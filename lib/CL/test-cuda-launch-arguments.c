/* Real parameter helpers with explicit host copy fixture; no CUDA/CL API calls. */
#include "pocl_cl.h"
#include "pocl_debug.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "pocl-cuda-kernel-data.h"
#include "pocl-cuda-local-layout.h"
#include "pocl-cuda-launch-types.h"
#include "pocl-cuda-launch-arguments.inc"

typedef struct
{
  struct _cl_kernel kernel;
  struct _cl_program program;
  struct _cl_device_id device;
  pocl_kernel_metadata_t meta;
  struct pocl_argument_info info[4];
  pocl_argument arguments[4];
  _cl_command_run run;
  pocl_cuda_kernel_data_t data;
  size_t alignments[5], automatic;
  void *params[64];
  unsigned local_offsets[8], constant_offsets[8];
  pocl_cuda_launch_arguments frame;
  struct _cl_mem memory[2];
  pocl_mem_identifier memory_slots[2];
  cl_mem handles[2];
  int scalar;
  CUdeviceptr raw;
} Fixture;

static void require (int condition, const char *message)
{
  if (!condition) { fprintf (stderr, "%s\n", message); exit (1); }
}

static void unexpected_copy (const pocl_cuda_constant_copy *copy)
{
  (void)copy;
  require (0, "unexpected constant copy fixture invocation");
}

static void fixture_copy (const pocl_cuda_constant_copy *copy)
{
  require (copy->stream == (CUstream)(uintptr_t)123, "copy stream changed");
  if (copy->destination == 1000)
    {
      require (copy->source == 10000, "first constant source");
      require (copy->bytes == 5, "first constant bytes");
      return;
    }
  require (copy->destination == 1008, "second constant padding");
  require (copy->source == 20000, "second constant source");
  require (copy->bytes == 4, "second constant bytes");
}

static void initialize (Fixture *fixture)
{
  memset (fixture, 0, sizeof (*fixture));
  fixture->kernel.program = &fixture->program;
  fixture->kernel.meta = &fixture->meta;
  fixture->run.kernel = &fixture->kernel;
  fixture->run.arguments = fixture->arguments;
  fixture->run.pc.work_dim = 3;
  fixture->run.global_size[0] = 128; fixture->run.global_size[1] = 6;
  fixture->run.global_size[2] = 5;
  fixture->run.pc.num_groups[0] = 2; fixture->run.pc.num_groups[1] = 3;
  fixture->run.pc.num_groups[2] = 5;
  fixture->run.pc.local_size[0] = 64; fixture->run.pc.local_size[1] = 2;
  fixture->run.pc.local_size[2] = 1;
  fixture->run.pc.global_offset[0] = 9; fixture->run.pc.global_offset[1] = 7;
  fixture->run.pc.global_offset[2] = 5;
  fixture->meta.arg_info = fixture->info;
  fixture->meta.local_sizes = &fixture->automatic;
  fixture->automatic = 28;
  fixture->data.alignments = fixture->alignments;
  fixture->scalar = 42; fixture->raw = 30000;
  for (unsigned i = 0; i < 2; ++i)
    {
      fixture->handles[i] = fixture->memory + i;
      fixture->memory[i].device_ptrs = fixture->memory_slots + i;
      fixture->memory_slots[i].mem_ptr = (void *)(uintptr_t)(10000 * (i + 1));
    }
  fixture->frame = (pocl_cuda_launch_arguments) {
    .params = fixture->params, .local_offsets = fixture->local_offsets,
    .constant_offsets = fixture->constant_offsets
  };
}

static pocl_cuda_argument_context context_for (Fixture *fixture)
{
  const pocl_cuda_argument_context result = {
    .run = &fixture->run, .device = &fixture->device,
    .stream = (CUstream)(uintptr_t)123,
    .target = { .kernel_data = &fixture->data },
    .copy_constant = unexpected_copy
  };
  return result;
}

static void prepare (Fixture *fixture, const pocl_cuda_argument_context *context)
{
  pocl_cuda_local_layout_input layout;
  require (pocl_cuda_launch_layout (context, &layout) == CL_SUCCESS,
           "valid launch layout rejected");
  fixture->frame.abi_count = layout.alignment_count;
  require (pocl_cuda_prepare_launch_arguments (context, &layout, &fixture->frame)
               == CL_SUCCESS, "valid parameter preparation rejected");
}

static void check_implicit (const Fixture *fixture)
{
  const pocl_cuda_launch_arguments *frame = &fixture->frame;
  size_t index = frame->abi_count;
  require (*(cl_uint *)frame->params[index++] == 3, "work_dim order/value");
  for (unsigned i = 0; i < 3; ++i)
    require (*(unsigned *)frame->params[index + i] == fixture->run.pc.global_offset[i],
             "global offset order/value");
  index += 3;
  for (unsigned i = 0; i < 3; ++i)
    require (*(unsigned *)frame->params[index + i] == fixture->run.global_size[i],
             "global size order/value");
  index += 3;
  for (unsigned i = 0; i < 3; ++i)
    require (*(unsigned *)frame->params[index + i] == fixture->run.pc.num_groups[i],
             "group count order/value");
  index += 3;
  for (unsigned i = 0; i < 3; ++i)
    require (*(unsigned *)frame->params[index + i] == fixture->run.pc.local_size[i],
             "enqueued size order/value");
  index += 3;
  for (unsigned i = 0; i < 3; ++i)
    require (frame->params[index + i] == frame->group_offsets + i,
             "region offsets must point at caller-owned live storage");
}

static void configure_locals (Fixture *fixture)
{
  fixture->meta.num_args = 3; fixture->meta.num_locals = 1;
  fixture->info[0] = (struct pocl_argument_info) {
    .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL };
  fixture->info[1] = (struct pocl_argument_info) {
    .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL };
  fixture->info[2] = fixture->info[1];
  fixture->arguments[0].value = &fixture->handles[0];
  fixture->arguments[1].size = 13; fixture->arguments[2].size = 512;
  fixture->alignments[0] = 4; fixture->alignments[1] = 1;
  fixture->alignments[2] = 8; fixture->alignments[3] = 4;
  fixture->data.alignment_capacity = 4;
}

static void test_explicit_and_automatic (void)
{
  Fixture f; initialize (&f); configure_locals (&f);
  const pocl_cuda_argument_context context = context_for (&f);
  prepare (&f, &context);
  require (f.frame.local_bytes == 556, "dynamic size must include auto once and padding");
  require (*(unsigned *)f.params[1] == 0, "tag offset");
  require (*(unsigned *)f.params[2] == 16, "double offset");
  require (*(unsigned *)f.params[3] == 528, "hidden automatic offset");
  require (f.params[0] == &f.memory_slots[0].mem_ptr, "global memory binding");
  require (f.meta.num_args == 3, "hidden argument changed user metadata");
  check_implicit (&f);
  f.frame.group_offsets[0] = 12;
  require (*(unsigned *)f.params[17] == 12, "region pointer lifetime");
}

static void test_nonlocal_and_static_auto (void)
{
  Fixture f; initialize (&f);
  f.meta.num_args = 4; f.meta.num_locals = 1;
  f.info[0].type = POCL_ARG_TYPE_NONE;
  for (unsigned i = 1; i < 4; ++i)
    f.info[i] = (struct pocl_argument_info) {
      .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL };
  f.arguments[0].value = &f.scalar;
  f.arguments[1].value = &f.handles[0];
  f.arguments[2].value = &f.raw; f.arguments[2].is_raw_ptr = 1;
  f.arguments[3].value = NULL;
  f.data.alignment_capacity = 4;
  const pocl_cuda_argument_context context = context_for (&f);
  prepare (&f, &context);
  require (f.frame.local_bytes == 0, "static automatic storage added dynamically");
  require (f.params[0] == &f.scalar, "scalar value pointer");
  require (f.params[1] == &f.memory_slots[0].mem_ptr, "global pointer address");
  require (f.params[2] == &f.raw, "raw pointer address");
  require (f.params[3] == &f.frame.null_pointer, "null global pointer storage");
  require (*(void **)f.params[3] == NULL, "null global pointer value");
  check_implicit (&f);
}

static void test_constant_bindings (void)
{
  Fixture f; initialize (&f);
  f.meta.num_args = 2; f.data.alignment_capacity = 2;
  for (unsigned i = 0; i < 2; ++i)
    {
      f.info[i] = (struct pocl_argument_info) {
        .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT };
      f.arguments[i].value = &f.handles[i];
    }
  f.memory[0].size = 5; f.memory[1].size = 4;
  f.alignments[0] = 4; f.alignments[1] = 8;
  pocl_cuda_argument_context context = context_for (&f);
  context.target.constant_base = 1000; context.target.constant_size = 64;
  context.copy_constant = fixture_copy;
  prepare (&f, &context);
  require (f.frame.constant_bytes == 12, "constant copy padding/count");
  require (*(unsigned *)f.params[0] == 0, "constant first offset binding");
  require (*(unsigned *)f.params[1] == 8, "constant second offset binding");
  require (f.frame.local_bytes == 0, "constant allocation polluted local bytes");
}

static void test_absent_constant_region (void)
{
  Fixture f; initialize (&f);
  f.meta.num_args = 1; f.data.alignment_capacity = 1;
  f.info[0] = (struct pocl_argument_info) {
    .type = POCL_ARG_TYPE_POINTER, .address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT };
  f.arguments[0].value = NULL;
  const pocl_cuda_argument_context context = context_for (&f);
  prepare (&f, &context);
  require (*(unsigned *)f.params[0] == 0, "absent constant offset");
  require (f.frame.constant_bytes == 0, "absent region copied memory");
}

static void test_invalid_metadata_and_overflow (void)
{
  Fixture f; initialize (&f); configure_locals (&f);
  pocl_cuda_argument_context context = context_for (&f);
  pocl_cuda_local_layout_input layout = { .alignment_count = 777 };
  f.data.alignment_capacity = 5;
  require (pocl_cuda_launch_layout (&context, &layout) == CL_INVALID_KERNEL,
           "compiled metadata/capacity mismatch accepted");
  require (layout.alignment_count == 777, "failed validation published layout");
  context.supplied_table = 1;
  require (pocl_cuda_launch_layout (&context, &layout) == CL_SUCCESS,
           "declared supplied-table prefix rejected");
  require (layout.alignment_count == 4, "supplied prefix has wrong active count");
  f.data.alignment_capacity = 3;
  require (pocl_cuda_launch_layout (&context, &layout) == CL_INVALID_KERNEL,
           "supplied table overread accepted");
  f.data.alignment_capacity = 4; context.supplied_table = 0;
  require (pocl_cuda_launch_layout (&context, &layout) == CL_SUCCESS, "restore layout");
  f.arguments[1].size = SIZE_MAX;
  f.frame.abi_count = 4;
  f.params[0] = (void *)(uintptr_t)0xfeed;
  require (pocl_cuda_prepare_launch_arguments (&context, &layout, &f.frame)
               == CL_OUT_OF_RESOURCES, "size overflow accepted");
  require (f.params[0] == (void *)(uintptr_t)0xfeed, "overflow bound a partial parameter");
}

static void test_zero_user_args (void)
{
  Fixture f; initialize (&f);
  f.meta.num_args = 0; f.meta.arg_info = NULL;
  f.run.arguments = NULL; f.data.alignment_capacity = 0; f.data.alignments = NULL;
  const pocl_cuda_argument_context context = context_for (&f);
  prepare (&f, &context);
  require (f.frame.abi_count == 0, "zero user arguments gained hidden locals");
  require (f.frame.local_bytes == 0, "zero local allocation rejected");
  check_implicit (&f);
}

int main (void)
{
  test_explicit_and_automatic ();
  test_nonlocal_and_static_auto ();
  test_constant_bindings ();
  test_absent_constant_region ();
  test_invalid_metadata_and_overflow ();
  test_zero_user_args ();
  puts ("CUDA launch preparation: 6 host groups passed; copy fixture only, no APIs");
  return 0;
}
