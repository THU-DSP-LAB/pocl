/* Host-only contract: explicit immutable callback fixtures, no CUDA/OpenCL API,
 * no device provider, no module compiler, and no claim of device execution. */
#include "pocl_cl.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK_ERROR(result, label) ((result) != CUDA_SUCCESS)
#include "pocl-cuda-resources.inc"

typedef struct
{
  const char *name;
  int shared;
  CUresult lookup_status;
  CUresult attribute_status;
} Fixture;

static void
require (int condition, const char *message)
{
  if (!condition)
    {
      fprintf (stderr, "%s\n", message);
      exit (1);
    }
}

static CUresult
fixture_function (CUfunction *output, CUmodule module, const char *name)
{
  const Fixture *records = (const Fixture *)module;
  for (unsigned i = 0; records[i].name != NULL; ++i)
    if (strcmp (records[i].name, name) == 0)
      {
        *output = (CUfunction)&records[i];
        return records[i].lookup_status;
      }
  return CUDA_ERROR_NOT_FOUND;
}

static CUresult
fixture_attribute (int *output, CUfunction_attribute attribute,
                    CUfunction function)
{
  if (attribute != CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    return CUDA_ERROR_INVALID_VALUE;
  const Fixture *record = (const Fixture *)function;
  *output = record->shared;
  return record->attribute_status;
}

static pocl_cuda_resource_api
fixture_api (const Fixture *records)
{
  const pocl_cuda_resource_api result = {
    (CUmodule)records, fixture_function, fixture_attribute
  };
  return result;
}

static void
test_late_scratch_and_repeat (void)
{
  const Fixture records[] = { { "scratch", 8192, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const pocl_cuda_resource_api api = fixture_api (records);
  cl_ulong local[2] = { 0, 123 };
  pocl_kernel_metadata_t meta = { .name = "scratch", .local_mem_size = local };
  struct _cl_program program = { .num_kernels = 1, .kernel_meta = &meta };
  for (int repeat = 0; repeat < 4; ++repeat)
    {
      require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_SUCCESS,
               "late scratch refresh failed");
      require (local[0] == 8192 && local[1] == 123,
               "late scratch missing, accumulated, or other device contaminated");
    }
}

static void
test_static_vs_automatic_dynamic (void)
{
  const Fixture records[] = { { "mixed", 64, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const pocl_cuda_resource_api api = fixture_api (records);
  size_t automatic[] = { 12, 20 };
  cl_ulong local[2] = { 888, 999 };
  struct pocl_argument_info argument = {
    .type = POCL_ARG_TYPE_POINTER,
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL
  };
  pocl_kernel_metadata_t meta = {
    .name = "mixed", .local_mem_size = local,
    .num_locals = 2, .local_sizes = automatic
  };
  struct _cl_program program = { .num_kernels = 1, .kernel_meta = &meta };
  require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_SUCCESS,
           "static automatic local refresh failed");
  require (local[0] == 64, "static automatic locals double counted");
  meta.num_args = 1;
  meta.arg_info = &argument;
  require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_SUCCESS,
           "dynamic automatic local refresh failed");
  require (local[0] == 64, "static base must not include automatic dynamic locals");
  require (local[1] == 999, "dynamic calculation contaminated another device");
  require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_SUCCESS,
           "repeated dynamic refresh failed");
  require (local[0] == 64, "repeated dynamic refresh accumulated previous base");
}

static void
test_transactional_errors (void)
{
  cl_ulong locals[2][2] = { { 111, 112 }, { 221, 222 } };
  pocl_kernel_metadata_t metas[2] = {
    { .name = "first", .local_mem_size = locals[0] },
    { .name = "second", .local_mem_size = locals[1] }
  };
  struct _cl_program program = { .num_kernels = 2, .kernel_meta = metas };
  const Fixture cases[][3] = {
    { { "first", 8, CUDA_SUCCESS, CUDA_SUCCESS },
      { "second", 16, CUDA_ERROR_INVALID_HANDLE, CUDA_SUCCESS }, { 0 } },
    { { "first", 8, CUDA_SUCCESS, CUDA_SUCCESS },
      { "second", 16, CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED }, { 0 } },
    { { "first", 8, CUDA_SUCCESS, CUDA_SUCCESS },
      { "second", -1, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } }
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
      const pocl_cuda_resource_api api = fixture_api (cases[i]);
      require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_OUT_OF_RESOURCES,
               "lookup/attribute/negative error silently accepted");
      require (locals[0][0] == 111 && locals[1][0] == 221,
               "failed batch partially published metadata");
      require (locals[0][1] == 112 && locals[1][1] == 222,
               "failed batch changed other device metadata");
    }
}

static void
test_zero_rebuild_and_device_slots (void)
{
  const Fixture zero[] = { { "zero", 0, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const Fixture other[] = { { "zero", 36, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const pocl_cuda_resource_api zero_api = fixture_api (zero);
  const pocl_cuda_resource_api other_api = fixture_api (other);
  cl_ulong local[2] = { 8192, 555 };
  pocl_kernel_metadata_t meta = { .name = "zero", .local_mem_size = local };
  struct _cl_program program = { .num_kernels = 1, .kernel_meta = &meta };
  require (pocl_cuda_refresh_local_sizes (&program, 0, &zero_api) == CL_SUCCESS,
           "legal zero resource rejected");
  require (local[0] == 0 && local[1] == 555, "rebuild retained obsolete static bytes");
  require (pocl_cuda_refresh_local_sizes (&program, 1, &other_api) == CL_SUCCESS,
           "second device refresh rejected");
  require (local[0] == 0 && local[1] == 36, "per-device metadata cross contaminated");
}

static void
test_invalid_metadata_and_overflow (void)
{
  const Fixture records[] = { { "invalid", 8, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const pocl_cuda_resource_api api = fixture_api (records);
  cl_ulong local = 777;
  pocl_kernel_metadata_t meta = { .name = "invalid", .local_mem_size = &local };
  struct _cl_program program = { .num_kernels = 1, .kernel_meta = &meta };
  meta.local_mem_size = NULL;
  require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_INVALID_PROGRAM,
           "missing associated devices accepted");
}

static void
test_binary_zero_metadata_slots (void)
{
  const Fixture records[] = { { "zero", 0, CUDA_SUCCESS, CUDA_SUCCESS }, { 0 } };
  const pocl_cuda_resource_api api = fixture_api (records);
  pocl_kernel_metadata_t meta = { .name = "zero", .local_mem_size = NULL };
  struct _cl_program program = {
    .num_kernels = 1, .kernel_meta = &meta, .associated_num_devices = 2
  };
  require (pocl_cuda_refresh_local_sizes (&program, 0, &api) == CL_SUCCESS,
           "valid zero-resource binary metadata rejected");
  require (meta.local_mem_size != NULL && meta.local_mem_size[0] == 0
               && meta.local_mem_size[1] == 0,
           "binary resource storage missing or another device contaminated");
  free (meta.local_mem_size);
}

static void *
reject_allocation (size_t count, size_t size)
{
  (void)count;
  (void)size;
  return NULL;
}

static void
test_binary_storage_allocation_failure (void)
{
  pocl_kernel_metadata_t meta = { .name = "zero", .local_mem_size = NULL };
  struct _cl_program program = {
    .num_kernels = 1, .kernel_meta = &meta, .associated_num_devices = 2
  };
  require (pocl_cuda_ensure_local_storage (&program, reject_allocation)
               == CL_OUT_OF_HOST_MEMORY,
           "real allocation failure silently accepted");
  require (meta.local_mem_size == NULL,
           "failed allocation published invalid storage");
}

int
main (void)
{
  test_late_scratch_and_repeat ();
  test_static_vs_automatic_dynamic ();
  test_transactional_errors ();
  test_zero_rebuild_and_device_slots ();
  test_invalid_metadata_and_overflow ();
  test_binary_zero_metadata_slots ();
  test_binary_storage_allocation_failure ();
  puts ("CUDA local resource host contracts: 7 groups passed (fixture callbacks only)");
  return 0;
}
