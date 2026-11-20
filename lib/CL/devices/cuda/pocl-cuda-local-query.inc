/* Use the launch layout with immutable static program resource metadata. */

cl_int
pocl_cuda_get_kernel_local_mem_size (cl_kernel kernel, unsigned device_i,
                                    cl_ulong *size)
{
  const pocl_kernel_metadata_t *meta = kernel->meta;
  if (!meta->data || !meta->data[device_i] || !meta->local_mem_size)
    return CL_INVALID_KERNEL;
  const pocl_cuda_kernel_data_t *kdata = meta->data[device_i];
  const pocl_cuda_local_layout_input input = {
    meta, kernel->dyn_arguments, kdata->alignments,
    kdata->alignment_capacity
  };
  unsigned dynamic_bytes = 0;
  cl_int result = pocl_cuda_plan_local_layout (&input, NULL, &dynamic_bytes);
  if (result != CL_SUCCESS)
    return result;
  if (dynamic_bytes > UINT64_MAX - meta->local_mem_size[device_i])
    return CL_OUT_OF_RESOURCES;
  *size = meta->local_mem_size[device_i] + dynamic_bytes;
  return CL_SUCCESS;
}
