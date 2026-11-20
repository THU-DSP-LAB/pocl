/* Compiled per-device kernel metadata is published only after ABI validation. */
int
pocl_cuda_compile_kernel (_cl_command_node *cmd,
                          cl_kernel kernel,
                          cl_device_id device,
                          int specialize)
{
  return CL_SUCCESS;
}

static cl_int
pocl_cuda_load_kernel_alignments (const pocl_kernel_metadata_t *meta,
                                  const pocl_cuda_program_data_t *pdata,
                                  pocl_cuda_kernel_data_t *kdata)
{
  size_t count = pocl_cuda_local_abi_count (meta);
  kdata->alignments = calloc (count ? count : 1, sizeof (size_t));
  if (!kdata->alignments)
    return CL_OUT_OF_HOST_MEMORY;
  const pocl_cuda_alignment_query_t query = {
    .kernel_name = meta->name, .alignments = kdata->alignments,
    .alignment_map = pdata->align_map_offsets, .num_alignments = count
  };
  if (pocl_cuda_get_ptr_arg_alignment (&query))
    {
      POCL_MEM_FREE (kdata->alignments);
      return CL_INVALID_KERNEL;
    }
  kdata->alignment_capacity = count;
  return CL_SUCCESS;
}

typedef struct
{
  CUfunction function;
  size_t max_threads;
} pocl_cuda_kernel_limits;

static cl_int
pocl_cuda_read_kernel_limits (CUmodule module, const char *name,
                              pocl_cuda_kernel_limits *limits)
{
  CUfunction function = NULL;
  CUresult result = cuModuleGetFunction (&function, module, name);
  if (CUDA_CHECK_ERROR (result, "cuModuleGetFunction"))
    return CL_OUT_OF_RESOURCES;
  int threads = 0;
  result = cuFuncGetAttribute (&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               function);
  if (CUDA_CHECK_ERROR (result, "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)"))
    return CL_OUT_OF_RESOURCES;
  if (threads <= 0)
    return CL_OUT_OF_RESOURCES;
  limits->function = function;
  limits->max_threads = threads;
  return CL_SUCCESS;
}

static void
pocl_cuda_set_kernel_limit (pocl_kernel_metadata_t *meta, unsigned device_i,
                            size_t max_threads)
{
  size_t limit = meta->max_workgroup_size[device_i];
  meta->max_workgroup_size[device_i]
      = limit == 0 ? max_threads : min (limit, max_threads);
}

static cl_int
pocl_cuda_create_kernel_data (cl_program program, cl_kernel kernel,
                              unsigned device_i)
{
  pocl_kernel_metadata_t *meta = kernel->meta;
  const pocl_cuda_local_layout_input metadata_input = {
    meta, kernel->dyn_arguments, NULL, 0
  };
  if (pocl_cuda_validate_local_layout (&metadata_input) != CL_SUCCESS)
    return CL_INVALID_KERNEL;
  pocl_cuda_program_data_t *pdata = program->data[device_i];
  if (!pdata || !meta->data)
    return CL_INVALID_PROGRAM_EXECUTABLE;
  pocl_cuda_kernel_data_t *kdata = meta->data[device_i];
  if (kdata != NULL)
    {
      ++kdata->refcount;
      return CL_SUCCESS;
    }
  pocl_cuda_kernel_limits limits;
  cl_int result = pocl_cuda_read_kernel_limits (pdata->module_offsets,
                                               kernel->name, &limits);
  if (result != CL_SUCCESS)
    return result;
  kdata = calloc (1, sizeof (*kdata));
  if (!kdata)
    return CL_OUT_OF_HOST_MEMORY;
  kdata->kernel_offsets = limits.function;
  if (!meta->max_workgroup_size)
    meta->max_workgroup_size = calloc (program->associated_num_devices,
                                       sizeof (size_t));
  if (!meta->max_workgroup_size)
    {
      free (kdata);
      return CL_OUT_OF_HOST_MEMORY;
    }
  result = pocl_cuda_load_kernel_alignments (meta, pdata, kdata);
  if (result != CL_SUCCESS)
    {
      free (kdata);
      return result;
    }
  pocl_cuda_set_kernel_limit (meta, device_i, limits.max_threads);
  kdata->refcount = 1;
  meta->data[device_i] = kdata;
  return CL_SUCCESS;
}

/* Existing device-ops ABI requires these four callback parameters. */
int
pocl_cuda_create_kernel (cl_device_id device, cl_program program,
                         cl_kernel kernel, unsigned device_i)
{
  return pocl_cuda_create_kernel_data (program, kernel, device_i);
}

int
pocl_cuda_free_kernel (cl_device_id device, cl_program program,
                       cl_kernel kernel, unsigned device_i)
{
  pocl_kernel_metadata_t *meta = kernel->meta;
  pocl_cuda_kernel_data_t *kdata
      = (pocl_cuda_kernel_data_t *)meta->data[device_i];
  if (kdata != NULL)
    {
      --kdata->refcount;
      if (kdata->refcount == 0)
        {
          POCL_MEM_FREE (kdata->alignments);
          POCL_MEM_FREE (kdata);
          meta->data[device_i] = NULL;
        }
    }
  return CL_SUCCESS;
}
