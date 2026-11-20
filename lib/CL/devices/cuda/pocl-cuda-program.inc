static int
pocl_cuda_build_cuda_builtins (cl_program program, cl_uint device_i)
{
  POCL_MSG_PRINT_CUDA ("preparing CUDA builtin kernels\n");
  cl_device_id dev = program->devices[device_i];
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)dev->data;
  /* these only need to be built once per device */
  if (ddata->cuda_builtin_kernels_built)
    {
      POCL_MSG_PRINT_CUDA ("CUDA builtin kernels already built\n");
      return 0;
    }

  int have_sm70 = (ddata->sm >= 70);

  uint64_t builtins_file_len = 0;
  char *builtins_file = NULL;
  char builtin_path[POCL_MAX_PATHNAME_LENGTH];

  char filename[64];
  filename[0] = '/';
  pocl_str_tolower (filename + 1, dev->ops->device_name);
  strcat (filename, "/");
  if (have_sm70)
    strcat (filename, "builtins_sm70.ptx");
  else
    strcat (filename, "builtins_sm50.ptx");

  pocl_get_srcdir_or_datadir (builtin_path, "/lib/CL/devices", "", filename);

  if (pocl_read_file (builtin_path, &builtins_file, &builtins_file_len) < 0)
    {
      POCL_MSG_ERR ("can't read cuda builtins from file %s\n", builtin_path);
      return -1;
    }

  CUresult res;
  CUfunction ff;
  CUmodule mod;
  res = cuModuleLoadData (&mod, builtins_file);
  if (CUDA_CHECK_ERROR (res, "cuModuleLoadData CudaBuiltinKernelsPTX"))
    return -1;

  memset (&ddata->cuda_builtin_kernels_program, 0,
          sizeof (pocl_cuda_program_data_t));
  ddata->cuda_builtin_kernels_program.module = mod;
  ddata->cuda_builtin_kernels_program.module_offsets = mod; // TODO fix this

  static size_t cuda_builtin_kernel_zero_alignments[20] = { 0 };

  res = cuModuleGetFunction (&ff, mod, "pocl_mul_i32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_mul_i32"))
    return -1;
  ddata->cuda_builtin_kernels_data[0].kernel = ff;
  ddata->cuda_builtin_kernels_data[0].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[0].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[0].alignment_capacity
      = sizeof (cuda_builtin_kernel_zero_alignments) / sizeof (cuda_builtin_kernel_zero_alignments[0]);
  ddata->cuda_builtin_kernels_data[0].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_add_i32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_add_i32"))
    return -1;
  ddata->cuda_builtin_kernels_data[1].kernel = ff;
  ddata->cuda_builtin_kernels_data[1].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[1].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[1].alignment_capacity
      = sizeof (cuda_builtin_kernel_zero_alignments) / sizeof (cuda_builtin_kernel_zero_alignments[0]);
  ddata->cuda_builtin_kernels_data[1].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_dnn_conv2d_int8_relu");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_dnn_conv2d_int8_relu"))
    return -1;
  ddata->cuda_builtin_kernels_data[2].kernel = ff;
  ddata->cuda_builtin_kernels_data[2].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[2].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[2].alignment_capacity
      = sizeof (cuda_builtin_kernel_zero_alignments) / sizeof (cuda_builtin_kernel_zero_alignments[0]);
  ddata->cuda_builtin_kernels_data[2].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_local_f32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_sgemm_local_f32"))
    return -1;
  ddata->cuda_builtin_kernels_data[3].kernel = ff;
  ddata->cuda_builtin_kernels_data[3].kernel_offsets = ff; // TODO fix this
  // 3 pointers, 3 unsigned
  static size_t sgemm_local_alignments[] = {0, 0, 0, 4, 4, 4,};
  ddata->cuda_builtin_kernels_data[3].alignments = sgemm_local_alignments;
  ddata->cuda_builtin_kernels_data[3].alignment_capacity
      = sizeof (sgemm_local_alignments) / sizeof (sgemm_local_alignments[0]);
  ddata->cuda_builtin_kernels_data[3].refcount = 1;

  if (have_sm70)
  {
    res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_tensor_f16f16f32");
    if (CUDA_CHECK_ERROR (res,
                          "cuModuleGetFunction  pocl_sgemm_tensor_f16f16f32"))
      return -1;
    ddata->cuda_builtin_kernels_data[4].kernel = ff;
    ddata->cuda_builtin_kernels_data[4].kernel_offsets = ff; // TODO fix this
    // 3 pointers, 3 unsigned
    static size_t sgemm_tensor_alignments[] = { 0, 0, 0, 4, 4, 4 };
    ddata->cuda_builtin_kernels_data[4].alignments = sgemm_tensor_alignments;
    ddata->cuda_builtin_kernels_data[4].alignment_capacity
      = sizeof (sgemm_tensor_alignments) / sizeof (sgemm_tensor_alignments[0]);
  ddata->cuda_builtin_kernels_data[4].refcount = 1;

    res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_scale_tensor_f16f16f32");
    if (CUDA_CHECK_ERROR (
            res, "cuModuleGetFunction  pocl_sgemm_scale_tensor_f16f16f32"))
      return -1;
    ddata->cuda_builtin_kernels_data[5].kernel = ff;
    ddata->cuda_builtin_kernels_data[5].kernel_offsets = ff; // TODO fix this
    // 3 pointers, 3 unsigned, 2 floats
    static size_t sgemm_tensor_scale_alignments[] = { 0, 0, 0, 4, 4, 4, 4, 4 };
    ddata->cuda_builtin_kernels_data[5].alignments
        = sgemm_tensor_scale_alignments;
    ddata->cuda_builtin_kernels_data[5].alignment_capacity
      = sizeof (sgemm_tensor_scale_alignments) / sizeof (sgemm_tensor_scale_alignments[0]);
  ddata->cuda_builtin_kernels_data[5].refcount = 1;
  }
  ddata->cuda_builtin_kernels_built = CL_TRUE;
  return 0;
}


int
pocl_cuda_build_builtin (cl_program program, cl_uint device_i)
{
  if (pocl_cuda_build_cuda_builtins (program, device_i) != 0)
    {
      POCL_MSG_ERR ("pocl-cuda: failed to build CUDA builtin kernels\n");
      return -1;
    }

  if (pocl_driver_build_opencl_builtins (program, device_i) != 0)
    {
      POCL_MSG_ERR ("pocl-cuda: failed to build OpenCL builtin kernels\n");
      return -1;
    }

  return 0;
}

static int
pocl_cuda_build_ptx (void *llvm_ir, cl_program prog, char *out_ptx, CUmodule *out_module,
                     cl_device_id device, pocl_cuda_device_data_t *ddata,
                     int use_offsets, CUdeviceptr *constant_mem_base,
                     size_t *constant_mem_size, void **alignments)
{
  assert (prog);
  assert (llvm_ir);
  assert (out_ptx);
  assert (out_module);
  CUresult result;

  /* Generate PTX from LLVM bitcode */
  if (!pocl_exists (out_ptx))
    {
      const pocl_ptx_gen_options_t options = {
        .device = device,
        .program = prog,
        .llvm_module = llvm_ir,
        .ptx_filename = out_ptx,
        .ptx_version = ddata->ptx,
        .libdevice_path = ddata->libdevice,
        .alignment_map = alignments,
      };
      int r = pocl_ptx_gen (&options);
      POCL_RETURN_ERROR_ON ((r != 0), CL_BUILD_PROGRAM_FAILURE,
                            "pocl-cuda: failed to generate PTX file %s\n",
                            out_ptx);
    }
  else
    {
      int r = pocl_cuda_create_alignments (llvm_ir, prog, alignments);
      POCL_RETURN_ERROR_ON (
          (r != 0), CL_BUILD_PROGRAM_FAILURE,
          "pocl-cuda: failed to create Alignments from BC\n");
    }

#ifdef POCL_DEBUG_MESSAGES
  if (!(pocl_debug_messages_filter & POCL_DEBUG_FLAG_CUDA))
    {
      result = cuModuleLoad (out_module, out_ptx);
      POCL_RETURN_ERROR_ON ((result != CUDA_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                            "cuModuleLoad PTX failed\n");
    }
  else
    {
#endif
      uintptr_t log_size = 1 << 12;
      char *log = (char *)malloc (log_size);
      CUjit_option opt[]
          = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
      char *content = NULL;
      uint64_t content_size = 0;

      pocl_read_file (out_ptx, &content, &content_size);
      POCL_RETURN_ERROR_ON ((content_size == 0), CL_BUILD_PROGRAM_FAILURE,
                            "failed to read PTX file: %s\n", out_ptx);
      void *val[] = { log, (void *)log_size };
      result = cuModuleLoadDataEx (out_module, content,
                                   sizeof (opt) / sizeof (opt[0]), opt, val);

      uintptr_t out_size = (uintptr_t)val[1];

      if (out_size > 0 || result != CUDA_SUCCESS)
        POCL_MSG_PRINT_CUDA ("cuModuleLoadDataEx(%s) log: %s\n", out_ptx, log);

      free (content);
      free (log);

      POCL_RETURN_ERROR_ON ((result != CUDA_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                            "cuModuleLoadDataEx failed\n");
#ifdef POCL_DEBUG_MESSAGES
    }
#endif

  /* LLVM removes the region when every constant pointer is unused. */
  *constant_mem_base = 0;
  *constant_mem_size = 0;
  result = cuModuleGetGlobal (constant_mem_base, constant_mem_size, *out_module,
                              "_constant_memory_region_");
  if (result == CUDA_ERROR_NOT_FOUND)
    {
      *constant_mem_base = 0;
      *constant_mem_size = 0;
    }
  POCL_RETURN_ERROR_ON (
      (result != CUDA_SUCCESS && result != CUDA_ERROR_NOT_FOUND),
      CL_BUILD_PROGRAM_FAILURE, "cuModuleGetGlobal constant region failed\n");

  return CL_SUCCESS;
}

static cl_int
pocl_cuda_publish_program_data (cl_program program, unsigned device_i,
                                const pocl_cuda_program_data_t *candidate)
{
  pocl_cuda_program_data_t *pdata = calloc (1, sizeof (*pdata));
  if (!pdata)
    return CL_OUT_OF_HOST_MEMORY;
  const pocl_cuda_resource_api api = {
    candidate->module_offsets, cuModuleGetFunction, cuFuncGetAttribute
  };
  cl_int result = pocl_cuda_refresh_local_sizes (program, device_i, &api);
  if (result != CL_SUCCESS)
    {
      free (pdata);
      return result;
    }
  *pdata = *candidate;
  program->data[device_i] = pdata;
  return CL_SUCCESS;
}

static cl_int
pocl_cuda_post_build_locked (cl_program program, unsigned device_i,
                             pocl_cuda_device_data_t *ddata)
{
  cl_device_id device = program->devices[device_i];
  cl_int result;
  char program_bc[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_bc_path (program_bc, program, device_i);

  char ofs_ptx_filename[POCL_MAX_PATHNAME_LENGTH];
  strcpy (ofs_ptx_filename, program_bc);
  strncat (ofs_ptx_filename, ".Ofs.ptx", POCL_MAX_PATHNAME_LENGTH - 1);

  /* Load PTX module */
  size_t constant_mem_size = 0;
  void *align_map = NULL;
  CUdeviceptr constant_mem_base;
  CUmodule ofs_module = NULL;

  assert (program->llvm_irs[device_i]);
  result = pocl_cuda_define_sub_group_size (
      program->llvm_irs[device_i], program, ddata->warp_size);
  if (result != CL_SUCCESS)
    goto ERROR_CLEANUP;
  result = pocl_llvm_run_passes_on_program (program, device_i);
  if (result != CL_SUCCESS)
    goto ERROR_CLEANUP;

  result = pocl_cuda_build_ptx (
      program->llvm_irs[device_i], program, ofs_ptx_filename, &ofs_module, device,
      ddata, 1, &constant_mem_base, &constant_mem_size, &align_map);
  if (result != CL_SUCCESS)
    goto ERROR_CLEANUP;
  const pocl_cuda_program_data_t candidate = {
    .module_offsets = ofs_module,
    .constant_mem_base_offsets = constant_mem_base,
    .constant_mem_size_offsets = constant_mem_size,
    .align_map_offsets = align_map
  };
  result = pocl_cuda_publish_program_data (program, device_i, &candidate);
  if (result != CL_SUCCESS)
    goto ERROR_CLEANUP;
  return result;
ERROR_CLEANUP:
  if (ofs_module != NULL)
    CUDA_CHECK_ERROR (cuModuleUnload (ofs_module),
                      "cuModuleUnload after program/resource failure");
  pocl_cuda_destroy_alignments (program->llvm_irs[device_i], align_map);
  return result;
}

int
pocl_cuda_post_build_program (cl_program program, cl_uint device_i)
{
  cl_device_id device = program->devices[device_i];
  pocl_cuda_device_data_t *ddata = device->data;
  if (CUDA_CHECK_ERROR (cuCtxSetCurrent (ddata->context), "cuCtxSetCurrent"))
    return CL_OUT_OF_RESOURCES;
  POCL_LOCK (ddata->compile_lock);
  cl_int result = pocl_cuda_post_build_locked (program, device_i, ddata);
  POCL_UNLOCK (ddata->compile_lock);
  return result;
}

int
pocl_cuda_free_program (cl_device_id device, cl_program program,
                        unsigned device_i)
{
  pocl_driver_free_program (device, program, device_i);

  CUresult result;
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;
  if (program->data[device_i] == NULL)
    return CL_SUCCESS;

  pocl_cuda_program_data_t *pdata
      = (pocl_cuda_program_data_t *)program->data[device_i];
  /*
      if (pdata->module)
      {
        result = cuModuleUnload(pdata->module);
        CUDA_CHECK (result, "cuModuleUnload");
        pdata->module = NULL;
      }
  */
  if (pdata->module_offsets)
    {
      result = cuModuleUnload (pdata->module_offsets);
      if (result != CUDA_SUCCESS)
        {
          POCL_MSG_ERR ("cuModuleUnload failed\n");
        }
      pdata->module_offsets = NULL;
    }
  if (pdata->align_map_offsets)
    {
      pocl_cuda_destroy_alignments (program->llvm_irs[device_i],
                                    pdata->align_map_offsets);
      pdata->align_map_offsets = NULL;
    }
  POCL_MEM_FREE (program->data[device_i]);
  return CL_SUCCESS;
}
