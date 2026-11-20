/* Normal API adapters and the stack-storage lifetime of one launch. */
static void
pocl_cuda_copy_constant_normal (const pocl_cuda_constant_copy *copy)
{
  CUresult result = cuMemcpyDtoDAsync (copy->destination, copy->source,
                                      copy->bytes, copy->stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoDAsync");
}

static void
pocl_cuda_launch_layout_error (cl_int status)
{
  POCL_ABORT ("[CUDA] invalid local argument ABI (%d)\n", status);
}

static pocl_cuda_local_layout_input
pocl_cuda_require_launch_layout (const pocl_cuda_argument_context *context)
{
  pocl_cuda_local_layout_input result;
  cl_int status = pocl_cuda_launch_layout (context, &result);
  if (status != CL_SUCCESS)
    pocl_cuda_launch_layout_error (status);
  return result;
}

static void
pocl_cuda_require_launch_arguments (const pocl_cuda_argument_context *context,
                                    const pocl_cuda_local_layout_input *layout,
                                    pocl_cuda_launch_arguments *frame)
{
  cl_int status = pocl_cuda_prepare_launch_arguments (context, layout, frame);
  if (status != CL_SUCCESS)
    pocl_cuda_launch_layout_error (status);
}

static void
pocl_cuda_launch_regions (const pocl_cuda_argument_context *context,
                          pocl_cuda_launch_arguments *frame)
{
  pocl_cuda_launch_region_t regions[POCL_CUDA_MAX_PARTITIONED_LAUNCHES];
  unsigned count = pocl_cuda_partition_launch (context->run, regions);
  for (unsigned i = 0; i < count; ++i)
    {
      memcpy (frame->group_offsets, regions[i].group_offset,
              sizeof (frame->group_offsets));
      CUresult result = cuLaunchKernel (
          context->target.function, regions[i].grid[0], regions[i].grid[1],
          regions[i].grid[2], regions[i].block[0], regions[i].block[1],
          regions[i].block[2], frame->local_bytes, context->stream,
          frame->params, NULL);
      CUDA_CHECK_ABORT (result, "cuLaunchKernel");
    }
}

static int
pocl_cuda_launch_has_work (const _cl_command_run *run)
{
  return run->pc.num_groups[0] != 0 && run->pc.num_groups[1] != 0
      && run->pc.num_groups[2] != 0;
}

static void
pocl_cuda_launch (const pocl_cuda_launch_inputs *input)
{
  const _cl_command_run *run = &input->command->command.run;
  if (!pocl_cuda_launch_has_work (run))
    return;
  pocl_cuda_launch_target target = { 0 };
  if (pocl_cuda_resolve_launch_target (input, &target) == POCL_CUDA_TARGET_HANDLED)
    return;
  const pocl_cuda_argument_context context = {
    run, input->device, input->stream, target, pocl_cuda_copy_constant_normal,
    run->kernel->program->num_builtin_kernels > 0
  };
  const pocl_cuda_local_layout_input layout = pocl_cuda_require_launch_layout (&context);
  const size_t count = layout.alignment_count;
  /* alloca storage must stay in this frame until all region API calls finish. */
  void **params = alloca (sizeof (void *) * (count + POCL_CUDA_IMPLICIT_PARAMETER_COUNT));
  unsigned *locals = alloca (sizeof (unsigned) * (count ? count : 1));
  unsigned *constants = alloca (sizeof (unsigned) * run->kernel->meta->num_args);
  pocl_cuda_launch_arguments frame = {
    .params = params, .local_offsets = locals, .constant_offsets = constants,
    .abi_count = count
  };
  pocl_cuda_require_launch_arguments (&context, &layout, &frame);
  pocl_cuda_launch_regions (&context, &frame);
}

/* Existing CUDA submit ABI remains a four-argument thin adapter. */
void
pocl_cuda_submit_kernel (CUstream stream, _cl_command_node *command,
                         cl_device_id device, cl_event event)
{
  const pocl_cuda_launch_inputs input = { stream, command, device, event };
  pocl_cuda_launch (&input);
}
