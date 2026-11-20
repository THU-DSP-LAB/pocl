/* Resolve the existing built-in or compiled entry before preparing arguments. */
enum
{
  POCL_CUDA_TARGET_MISSING,
  POCL_CUDA_TARGET_READY,
  POCL_CUDA_TARGET_HANDLED
};

static int
pocl_cuda_select_builtin_target (const pocl_cuda_launch_inputs *input,
                                 pocl_cuda_launch_target *target)
{
  pocl_cuda_device_data_t *data = input->device->data;
  const char *name = input->command->command.run.kernel->name;
  int found = POCL_CUDA_TARGET_MISSING;
  for (size_t i = 0; i < CUDA_BUILTIN_KERNELS; ++i)
    if (strcmp (name, cuda_builtin_kernels[i]) == 0)
      {
        *target = (pocl_cuda_launch_target) {
          data->cuda_builtin_kernels_data[i].kernel,
          &data->cuda_builtin_kernels_data[i],
          data->cuda_builtin_kernels_program.constant_mem_base,
          data->cuda_builtin_kernels_program.constant_mem_size
        };
        found = POCL_CUDA_TARGET_READY;
        break;
      }
#ifdef ENABLE_CUDNN
  for (size_t i = 0; i < CUDNN_BUILTIN_KERNELS; ++i)
    if (strcmp (name, cudnn_builtin_kernels[i]) == 0)
      {
        submit_cudnn_kernel (input->stream, input->command, input->device,
                             input->event);
        return POCL_CUDA_TARGET_HANDLED;
      }
#endif
  return found;
}

static void
pocl_cuda_select_compiled_target (const pocl_cuda_launch_inputs *input,
                                  pocl_cuda_launch_target *target)
{
  cl_kernel kernel = input->command->command.run.kernel;
  char *saved_name = NULL;
  for (size_t i = 0; i < OPENCL_BUILTIN_KERNELS; ++i)
    if (strcmp (kernel->name, opencl_builtin_kernels[i]) == 0)
      {
        pocl_sanitize_builtin_kernel_name (kernel, &saved_name);
        break;
      }
  unsigned index = input->command->program_device_i;
  const pocl_cuda_kernel_data_t *data = kernel->meta->data[index];
  const pocl_cuda_program_data_t *program = kernel->program->data[index];
  assert (data);
  *target = (pocl_cuda_launch_target) {
    data->kernel_offsets, data, program->constant_mem_base_offsets,
    program->constant_mem_size_offsets
  };
  if (saved_name)
    pocl_restore_builtin_kernel_name (kernel, saved_name);
}

static int
pocl_cuda_resolve_launch_target (const pocl_cuda_launch_inputs *input,
                                 pocl_cuda_launch_target *target)
{
  if (input->command->command.run.kernel->program->num_builtin_kernels > 0)
    {
      assert (0 && "builtin kernels unsupported");
      int result = pocl_cuda_select_builtin_target (input, target);
      if (result != POCL_CUDA_TARGET_MISSING)
        return result;
    }
  pocl_cuda_select_compiled_target (input, target);
  return POCL_CUDA_TARGET_READY;
}
