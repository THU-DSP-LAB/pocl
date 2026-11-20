/* Parameter binding shared by the production launch and finite host tests.
 * All pointed-to input metadata is borrowed; only caller-owned frame is built. */
static cl_int
pocl_cuda_launch_layout (const pocl_cuda_argument_context *context,
                         pocl_cuda_local_layout_input *result)
{
  const pocl_kernel_metadata_t *meta = context->run->kernel->meta;
  const pocl_cuda_kernel_data_t *data = context->target.kernel_data;
  size_t active = pocl_cuda_local_abi_count (meta);
  if (context->supplied_table && active > data->alignment_capacity)
    return CL_INVALID_KERNEL;
  const pocl_cuda_local_layout_input input = {
    meta, context->run->arguments, data->alignments,
    context->supplied_table ? active : data->alignment_capacity
  };
  cl_int status = pocl_cuda_validate_local_layout (&input);
  if (status != CL_SUCCESS)
    return status;
  if (!pocl_cuda_valid_alignment_count (&input))
    return CL_INVALID_KERNEL;
  *result = input;
  return CL_SUCCESS;
}

static void
pocl_cuda_copy_constant_argument (const pocl_cuda_argument_context *context,
                                  unsigned index,
                                  pocl_cuda_launch_arguments *frame)
{
  const pocl_argument *argument = &context->run->arguments[index];
  assert (argument->is_raw_ptr == 0);
  if (context->target.constant_size == 0)
    {
      assert (context->target.constant_base == 0);
      frame->constant_offsets[index] = 0;
      frame->params[index] = frame->constant_offsets + index;
      return;
    }
  assert (context->target.constant_base);
  cl_mem mem = *(void **)argument->value;
  CUdeviceptr source = (CUdeviceptr)mem->device_ptrs[context->device->global_mem_id].mem_ptr;
  size_t alignment = context->target.kernel_data->alignments[index];
  assert (alignment && "Zero alignment for pointer argument!");
  if (frame->constant_bytes % alignment)
    frame->constant_bytes += alignment - frame->constant_bytes % alignment;
  const pocl_cuda_constant_copy copy = {
    context->target.constant_base + frame->constant_bytes,
    source, mem->size, context->stream
  };
  context->copy_constant (&copy);
  frame->constant_offsets[index] = frame->constant_bytes;
  frame->params[index] = frame->constant_offsets + index;
  frame->constant_bytes += mem->size;
}

static void
pocl_cuda_bind_global_argument (const pocl_cuda_argument_context *context,
                                unsigned index,
                                pocl_cuda_launch_arguments *frame)
{
  const pocl_argument *argument = &context->run->arguments[index];
  assert (argument->is_raw_ptr == 0);
  if (argument->value)
    {
      cl_mem mem = *(void **)argument->value;
      frame->params[index] = &mem->device_ptrs[context->device->global_mem_id].mem_ptr;
    }
  else
    frame->params[index] = &frame->null_pointer;
}

static void
pocl_cuda_bind_pointer_argument (const pocl_cuda_argument_context *context,
                                 unsigned index,
                                 pocl_cuda_launch_arguments *frame)
{
  const pocl_kernel_metadata_t *meta = context->run->kernel->meta;
  const pocl_argument *argument = &context->run->arguments[index];
  if (ARG_IS_LOCAL (meta->arg_info[index]))
    frame->params[index] = frame->local_offsets + index;
  else if (argument->is_raw_ptr == 1)
    frame->params[index] = argument->value;
  else if (meta->arg_info[index].address_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT)
    pocl_cuda_copy_constant_argument (context, index, frame);
  else
    pocl_cuda_bind_global_argument (context, index, frame);
}

static void
pocl_cuda_unsupported_argument (void)
{
  POCL_ABORT ("Unhandled argument type for CUDA\n");
}

static void
pocl_cuda_bind_user_arguments (const pocl_cuda_argument_context *context,
                              pocl_cuda_launch_arguments *frame)
{
  const pocl_kernel_metadata_t *meta = context->run->kernel->meta;
  for (unsigned i = 0; i < meta->num_args; ++i)
    {
      switch (meta->arg_info[i].type)
        {
        case POCL_ARG_TYPE_NONE:
          frame->params[i] = context->run->arguments[i].value;
          break;
        case POCL_ARG_TYPE_POINTER:
          pocl_cuda_bind_pointer_argument (context, i, frame);
          break;
        case POCL_ARG_TYPE_IMAGE:
        case POCL_ARG_TYPE_SAMPLER:
        case POCL_ARG_TYPE_PIPE:
          pocl_cuda_unsupported_argument ();
          break;
        }
    }
}

static void
pocl_cuda_constant_limit_error (const pocl_cuda_argument_context *context,
                                const pocl_cuda_launch_arguments *frame)
{
  POCL_ABORT ("[CUDA] Total constant buffer size %u exceeds %lu allocated\n",
               frame->constant_bytes, context->target.constant_size);
}

static void
pocl_cuda_bind_automatic_locals (const pocl_cuda_argument_context *context,
                                pocl_cuda_launch_arguments *frame)
{
  size_t start = context->run->kernel->meta->num_args;
  for (size_t i = start; i < frame->abi_count; ++i)
    frame->params[i] = frame->local_offsets + i;
}

static void
pocl_cuda_copy_launch_dimensions (const pocl_cuda_argument_context *context,
                                  pocl_cuda_launch_arguments *frame)
{
  frame->work_dim = context->run->pc.work_dim;
  for (unsigned i = 0; i < 3; ++i)
    {
      assert (context->run->global_size[i] <= UINT_MAX);
      assert (context->run->pc.num_groups[i] <= UINT_MAX);
      assert (context->run->pc.local_size[i] <= UINT_MAX);
      frame->global_offsets[i] = context->run->pc.global_offset[i];
      frame->global_sizes[i] = context->run->global_size[i];
      frame->total_groups[i] = context->run->pc.num_groups[i];
      frame->enqueued_local_sizes[i] = context->run->pc.local_size[i];
    }
}

static size_t
pocl_cuda_bind_triplet (pocl_cuda_launch_arguments *frame, size_t index,
                        unsigned values[3])
{
  for (unsigned i = 0; i < 3; ++i)
    frame->params[index++] = values + i;
  return index;
}

static void
pocl_cuda_bind_implicit_arguments (const pocl_cuda_argument_context *context,
                                  pocl_cuda_launch_arguments *frame)
{
  pocl_cuda_copy_launch_dimensions (context, frame);
  size_t index = frame->abi_count;
  frame->params[index++] = &frame->work_dim;
  index = pocl_cuda_bind_triplet (frame, index, frame->global_offsets);
  index = pocl_cuda_bind_triplet (frame, index, frame->global_sizes);
  index = pocl_cuda_bind_triplet (frame, index, frame->total_groups);
  index = pocl_cuda_bind_triplet (frame, index, frame->enqueued_local_sizes);
  index = pocl_cuda_bind_triplet (frame, index, frame->group_offsets);
  assert (index == frame->abi_count + POCL_CUDA_IMPLICIT_PARAMETER_COUNT);
}

static cl_int
pocl_cuda_prepare_launch_arguments (const pocl_cuda_argument_context *context,
                                    const pocl_cuda_local_layout_input *layout,
                                    pocl_cuda_launch_arguments *frame)
{
  cl_int status = pocl_cuda_plan_local_layout (layout, frame->local_offsets,
                                             &frame->local_bytes);
  if (status != CL_SUCCESS)
    return status;
  pocl_cuda_bind_user_arguments (context, frame);
  if (frame->constant_bytes > context->target.constant_size)
    pocl_cuda_constant_limit_error (context, frame);
  pocl_cuda_bind_automatic_locals (context, frame);
  pocl_cuda_bind_implicit_arguments (context, frame);
  return CL_SUCCESS;
}
