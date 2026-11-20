/* Real launch/query layout helpers only; no Driver or OpenCL API calls. */
#include "pocl_cl.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "pocl-cuda-kernel-data.h"
#include "pocl-cuda-local-layout.h"
#include "pocl-cuda-local-query.inc"

static void require (int condition, const char *message)
{
  if (!condition) { fprintf (stderr, "%s\n", message); exit (1); }
}

static void test_padding_and_automatic (void)
{
  struct pocl_argument_info info[4] = {
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL },
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL },
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL },
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL }
  };
  pocl_argument args[4] = { { 0 }, { 0 }, { .size = 13 }, { .size = 512 } };
  size_t alignments[] = { 4, 4, 1, 8, 4 };
  size_t automatic = 28;
  pocl_kernel_metadata_t meta = { .num_args = 4, .arg_info = info,
    .num_locals = 1, .local_sizes = &automatic };
  pocl_cuda_local_layout_input input = { &meta, args, alignments, 5 };
  unsigned bytes = 777, offsets[5] = { 99, 99, 99, 99, 99 };
  require (pocl_cuda_plan_local_layout (&input, offsets, &bytes) == CL_SUCCESS,
           "valid mixed local layout rejected");
  require (bytes == 556 && offsets[2] == 0 && offsets[3] == 16 && offsets[4] == 528,
           "explicit or automatic padding/offset wrong");
  require (offsets[0] == 99 && offsets[1] == 99 && meta.num_args == 4,
           "layout touched global args or user arg count");
  meta.num_locals = 0; input.alignment_count = 4;
  require (pocl_cuda_plan_local_layout (&input, offsets, &bytes) == CL_SUCCESS,
           "explicit-only layout rejected");
  require (bytes == 528, "13+512 must include three alignment bytes");
}

static void test_query_repeats_and_device_slots (void)
{
  struct pocl_argument_info info[2] = {
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL },
    { .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL }
  };
  pocl_argument first[2] = { { .size = 13 }, { .size = 512 } };
  pocl_argument second[2] = { { .size = 17 }, { .size = 512 } };
  size_t alignments[] = { 1, 8 };
  pocl_cuda_kernel_data_t kdata = { .alignments = alignments, .alignment_capacity = 2 };
  void *data[] = { &kdata, &kdata };
  cl_ulong bases[] = { 8192, 4096 };
  pocl_kernel_metadata_t meta = { .num_args = 2, .arg_info = info,
    .data = data, .local_mem_size = bases };
  struct _cl_kernel kernel = { .meta = &meta, .dyn_arguments = first };
  struct _cl_kernel other = { .meta = &meta, .dyn_arguments = second };
  cl_ulong value = 777;
  for (unsigned repeat = 0; repeat < 4; ++repeat)
    {
      require (pocl_cuda_get_kernel_local_mem_size (&kernel, 0, &value) == CL_SUCCESS
               && value == 8720, "query lost padding or accumulated");
      require (pocl_cuda_get_kernel_local_mem_size (&other, 0, &value) == CL_SUCCESS
               && value == 8728, "shared metadata leaked per-kernel local args");
      require (pocl_cuda_get_kernel_local_mem_size (&kernel, 1, &value) == CL_SUCCESS
               && value == 4624, "query used wrong device base");
    }
  require (bases[0] == 8192 && bases[1] == 4096, "query mutated static metadata");
}

static void test_zero_and_static_automatic (void)
{
  size_t alignment = 4, automatic = 28;
  pocl_cuda_kernel_data_t kdata = { .alignments = &alignment };
  void *data[] = { &kdata };
  cl_ulong base = 0, value = 777;
  pocl_kernel_metadata_t meta = { .data = data, .local_mem_size = &base,
    .num_locals = 1, .local_sizes = &automatic };
  struct _cl_kernel kernel = { .meta = &meta };
  require (pocl_cuda_get_kernel_local_mem_size (&kernel, 0, &value) == CL_SUCCESS
           && value == 0, "legal zero resource rejected");
  base = 8192;
  require (pocl_cuda_get_kernel_local_mem_size (&kernel, 0, &value) == CL_SUCCESS
           && value == 8192, "static automatic locals double counted");
}

static void test_invalid_layout (void)
{
  struct pocl_argument_info info = { .type = POCL_ARG_TYPE_POINTER,
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL };
  pocl_argument args = { .size = 13 };
  size_t alignment = 8;
  pocl_kernel_metadata_t meta = { .num_args = 1, .arg_info = &info };
  pocl_cuda_local_layout_input input = { &meta, &args, &alignment, 1 };
  unsigned value = 777;
  input.alignment_count = 2;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_INVALID_KERNEL
           && value == 777, "ABI mismatch published layout");
  input.alignment_count = 1; alignment = 0;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_INVALID_KERNEL,
           "unknown zero alignment accepted");
  alignment = 3;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_INVALID_KERNEL,
           "non-power-of-two alignment accepted");
  alignment = 8; meta.num_locals = 1; input.alignment_count = 2;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_INVALID_KERNEL,
           "missing automatic sizes accepted");
  meta.num_locals = 0; input.alignment_count = 1; meta.arg_info = NULL;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_INVALID_KERNEL,
           "missing argument metadata accepted");
  require (value == 777, "failed layout changed byte count");
}

static void test_overflow (void)
{
  struct pocl_argument_info info = { .type = POCL_ARG_TYPE_POINTER,
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL };
  pocl_argument args = { .size = SIZE_MAX };
  size_t alignment = 8;
  pocl_kernel_metadata_t meta = { .num_args = 1, .arg_info = &info };
  pocl_cuda_local_layout_input input = { &meta, &args, &alignment, 1 };
  unsigned value = 777;
  require (pocl_cuda_plan_local_layout (&input, NULL, &value) == CL_OUT_OF_RESOURCES
           && value == 777, "dynamic size overflow wrapped or published");
}

static void test_query_bad_capacity (void)
{
  struct pocl_argument_info info = { .type = POCL_ARG_TYPE_POINTER,
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL };
  pocl_argument args = { .size = 13 };
  size_t alignment = 1;
  pocl_cuda_kernel_data_t kdata = { .alignments = &alignment, .alignment_capacity = 0 };
  void *data[] = { &kdata };
  cl_ulong base = 8, value = 777;
  pocl_kernel_metadata_t meta = { .num_args = 1, .arg_info = &info,
    .data = data, .local_mem_size = &base };
  struct _cl_kernel kernel = { .meta = &meta, .dyn_arguments = &args };
  require (pocl_cuda_get_kernel_local_mem_size (&kernel, 0, &value) == CL_INVALID_KERNEL
           && value == 777, "query ignored stored ABI capacity or changed output");
  kdata.alignment_capacity = 1;
  require (pocl_cuda_get_kernel_local_mem_size (&kernel, 0, &value) == CL_SUCCESS
           && value == 21, "valid capacity did not recover without metadata mutation");
}

int main (void)
{
  test_padding_and_automatic ();
  test_query_repeats_and_device_slots ();
  test_zero_and_static_automatic ();
  test_invalid_layout ();
  test_overflow ();
  test_query_bad_capacity ();
  puts ("CUDA shared local layout/query: 6 host groups passed; no API/device");
  return 0;
}
