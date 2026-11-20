/* Read final CUDA module resources after all LLVM/PTX scratch lowering.
 * The API table is explicit for finite host contract tests; production supplies
 * the normal CUDA Driver entry points, with no alternative provider. */
typedef struct
{
  CUmodule module;
  CUresult (*get_function) (CUfunction *, CUmodule, const char *);
  CUresult (*get_attribute) (int *, CUfunction_attribute, CUfunction);
} pocl_cuda_resource_api;

static cl_int
pocl_cuda_read_kernel_local_size (const pocl_kernel_metadata_t *meta,
                                  const pocl_cuda_resource_api *api,
                                  cl_ulong *local_size)
{
  CUfunction function = NULL;
  CUresult result = api->get_function (&function, api->module, meta->name);
  if (CUDA_CHECK_ERROR (result, "cuModuleGetFunction(local resource)"))
    return CL_OUT_OF_RESOURCES;
  int static_bytes = -1;
  result = api->get_attribute (&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                               function);
  if (CUDA_CHECK_ERROR (result, "cuFuncGetAttribute(SHARED_SIZE_BYTES)"))
    return CL_OUT_OF_RESOURCES;
  if (static_bytes < 0)
    return CL_OUT_OF_RESOURCES;

  /* Per-device program metadata stores only immutable final static bytes.
   * Per-kernel explicit and automatic dynamic locals are laid out at query. */
  *local_size = (cl_ulong)static_bytes;
  return CL_SUCCESS;
}

static cl_int
pocl_cuda_refresh_local_sizes (cl_program program, unsigned device_i,
                               const pocl_cuda_resource_api *api)
{
  cl_ulong *sizes = calloc (program->num_kernels, sizeof (cl_ulong));
  if (sizes == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  cl_int result = CL_SUCCESS;
  for (unsigned i = 0; i < program->num_kernels; ++i)
    {
      if (program->kernel_meta[i].local_mem_size == NULL)
        {
          result = CL_OUT_OF_HOST_MEMORY;
          break;
        }
      result = pocl_cuda_read_kernel_local_size (&program->kernel_meta[i], api,
                                                 &sizes[i]);
      if (result != CL_SUCCESS)
        break;
    }
  /* Commit once, only after every normal Driver query succeeds. Rebuilding or
   * recreating a kernel never accumulates an earlier static resource value. */
  if (result == CL_SUCCESS)
    for (unsigned i = 0; i < program->num_kernels; ++i)
      program->kernel_meta[i].local_mem_size[device_i] = sizes[i];
  free (sizes);
  return result;
}
