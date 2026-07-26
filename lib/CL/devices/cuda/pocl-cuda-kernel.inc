#ifdef ENABLE_CUDNN
void
submit_cudnn_kernel(CUstream stream, _cl_command_node *cmd,
                    cl_device_id device, cl_event event)
{

  _cl_command_run run = cmd->command.run;
  cl_kernel kernel = run.kernel;
  cl_program prog = kernel->program;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;
  pocl_kernel_metadata_t *meta = kernel->meta;

  // Input, weight and output buffers come from pocl's buffer management
  cl_mem mem = *(void **)arguments[0].value;
  float* in_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[0].offset);
  // cl_float16* in_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[0].offset);

  mem = *(void **)arguments[1].value;
  float* filt_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[1].offset);
  // cl_float16* filt_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[1].offset);

  mem = *(void **)arguments[2].value;
  float* out_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[2].offset);
  // cl_float16* out_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[2].offset);

  // All the other convolution dimensions are passed as arguments.
  int in_n      = *(int*)(arguments[3].value);
  int in_c      = *(int*)(arguments[4].value);
  int in_h      = *(int*)(arguments[5].value);
  int in_w      = *(int*)(arguments[6].value);

  int filt_k = *(int*)(arguments[7].value);
  int filt_c = *(int*)(arguments[8].value);
  int filt_h = *(int*)(arguments[9].value);
  int filt_w = *(int*)(arguments[10].value);

  int str_h = *(int*)(arguments[11].value);
  int str_w = *(int*)(arguments[12].value);
  int dil_h = *(int*)(arguments[13].value);
  int dil_w = *(int*)(arguments[14].value);
  int pad_h = *(int*)(arguments[15].value);
  int pad_w = *(int*)(arguments[16].value);

  int groups = *(int*)(arguments[17].value);

  float alpha = *(float*)(arguments[18].value);
  float beta  = *(float*)(arguments[19].value);

  /*POCL_MSG_PRINT_INFO("ARGS:%zx,%zx,%zx in:%i,%i,%i,%i filt %i,%i,%i,%i strdilpad %i,%i,%i,%i,%i,%i\n",
  in_data, filt_data, out_data, in_n, in_c, in_h, in_w, filt_k, filt_c, filt_h, filt_w,
   str_h, str_w, dil_h, dil_w,pad_h,pad_w);*/

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, /*CUDNN_DATA_HALF*/
        in_n, in_c, in_h, in_w));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT /*CUDNN_DATA_HALF*/, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT /*CUDNN_DATA_HALF*/));

  //For depth-wise convolutions
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));

  // output
  int out_n,out_c,out_h,out_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,/*CUDNN_DATA_HALF*/
        out_n, out_c, out_h, out_w));

  // algorithm
  cudnnConvolutionFwdAlgoPerf_t algos;
  int return_count = 0;
  /*CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        1, &return_count, &algos));

  cudnnConvolutionFwdAlgo_t algo = algos.algo;*/
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // try IMPLICIT_PRECOMP_GEMM if tensor cores not working
  POCL_MSG_PRINT_INFO("CuDNN Picking ALGO %d",algo);

  // CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH)); // this may not be necessary

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
  float *ws_data;
  CUDA_CALL(cudaMalloc((void **)&ws_data, ws_size));

  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));

 // Not sure if needed
  CUDA_CALL (cudaDeviceSynchronize ());

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
}

#endif //ENABLE_CUDNN

void
pocl_cuda_submit_kernel (CUstream stream, _cl_command_node *cmd,
                         cl_device_id device, cl_event event)
{
  _cl_command_run run = cmd->command.run;
  cl_kernel kernel = run.kernel;
  cl_program prog = kernel->program;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;
  pocl_kernel_metadata_t *meta = kernel->meta;
  CUdeviceptr constant_mem_base;
  size_t constant_mem_size;
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;

  if (pc.num_groups[0] == 0 || pc.num_groups[1] == 0 || pc.num_groups[2] == 0)
    return;

  /* Check if we need to handle global work offsets */
  int has_offsets =
    (pc.global_offset[0] || pc.global_offset[1] || pc.global_offset[2]);

  CUmodule module = NULL;
  CUfunction function = NULL;
  pocl_cuda_kernel_data_t *kdata = NULL;
  char *saved_kernel_name = NULL;

  /* Get kernel function */
  if (prog->num_builtin_kernels > 0)
    {
      assert (0 && "builtin kernels unsupported");
      /* CUDA builtins */
      for (size_t i = 0; i < CUDA_BUILTIN_KERNELS; ++i)
        {
          if (strcmp (kernel->name, cuda_builtin_kernels[i]) == 0)
            {
              module = ddata->cuda_builtin_kernels_program.module;
              kdata = &ddata->cuda_builtin_kernels_data[i];
              function = kdata->kernel;
              break;
            }
        }
#ifdef ENABLE_CUDNN
      /* CUDNN builtins */
      for (size_t i = 0; i < CUDNN_BUILTIN_KERNELS; ++i)
        {
          if (strcmp(kernel->name, cudnn_builtin_kernels[i]) == 0)
            {
              submit_cudnn_kernel(stream, cmd, device, event);
              return;
            }
        }
#endif //ENABLE_CUDNN
    }

  if (kdata == NULL)
    {
      /* handle OpenCL builtins */
      for (size_t i = 0; i < OPENCL_BUILTIN_KERNELS; ++i)
        {
          if (strcmp (kernel->name, opencl_builtin_kernels[i]) == 0)
            {
              pocl_sanitize_builtin_kernel_name (kernel, &saved_kernel_name);
              break;
            }
        }
      int has_offsets1 = 1;
      kdata = (pocl_cuda_kernel_data_t *)meta->data[cmd->program_device_i];
      pocl_cuda_program_data_t *pdata
          = (pocl_cuda_program_data_t *)prog->data[cmd->program_device_i];
      module = has_offsets1 ? pdata->module_offsets : pdata->module;
      function = has_offsets1 ? kdata->kernel_offsets : kdata->kernel;
      constant_mem_base = has_offsets1 ? pdata->constant_mem_base_offsets
                                       : pdata->constant_mem_base;
      constant_mem_size = has_offsets1 ? pdata->constant_mem_size_offsets
                                       : pdata->constant_mem_size;
      if (saved_kernel_name)
        pocl_restore_builtin_kernel_name (kernel, saved_kernel_name);
    }

  assert (kdata);

  /* Prepare kernel arguments */
  void *null = NULL;
  unsigned sharedMemBytes = 0;
  void **params
      = alloca (sizeof (void *) * (meta->num_args + meta->num_locals + 4));
  unsigned *sharedMemOffsets
      = alloca (sizeof (unsigned) * (meta->num_args + meta->num_locals));
  unsigned constantMemBytes = 0;
  unsigned *constantMemOffsets = alloca (sizeof (unsigned) * meta->num_args);
  unsigned globalOffsets[3];

  CUresult result;
  unsigned i;
  for (i = 0; i < meta->num_args; i++)
    {
      pocl_argument_type type = meta->arg_info[i].type;
      switch (type)
        {
        case POCL_ARG_TYPE_NONE:
          params[i] = arguments[i].value;
          break;
        case POCL_ARG_TYPE_POINTER:
          {
            if (ARG_IS_LOCAL (meta->arg_info[i]))
              {
                size_t size = arguments[i].size;
                size_t align = kdata->alignments[i];
                if (align < 1)
                  align = 1;
                if (size == 0)
                  params[i] = NULL;
                else {
                  /* Pad offset to align memory */
                  if (sharedMemBytes % align)
                    sharedMemBytes += align - (sharedMemBytes % align);

                  sharedMemOffsets[i] = sharedMemBytes;
                  params[i] = sharedMemOffsets + i;

                  sharedMemBytes += size;
                }
              }
            else if (arguments[i].is_raw_ptr == 1)
              {
                params[i] = arguments[i].value;
              }
            else if (meta->arg_info[i].address_qualifier
              == CL_KERNEL_ARG_ADDRESS_CONSTANT)
              {
                assert (arguments[i].is_raw_ptr == 0);

                /* An absent region means every constant pointer was removed
                 * as unused. The offset argument remains in the kernel ABI. */
                if (constant_mem_size == 0)
                  {
                    assert (constant_mem_base == 0);
                    constantMemOffsets[i] = 0;
                    params[i] = constantMemOffsets + i;
                    break;
                  }
                assert (constant_mem_base);

                /* Get device pointer */
                cl_mem mem = *(void **)arguments[i].value;
                CUdeviceptr src
                  = (CUdeviceptr)mem->device_ptrs[device->global_mem_id]
                      .mem_ptr;

                size_t align = kdata->alignments[i];
                assert (align && "Zero alignment for pointer argument!");
                if (constantMemBytes % align)
                  {
                    constantMemBytes += align - (constantMemBytes % align);
                  }

                /* Copy to constant buffer at current offset */
                result
                    = cuMemcpyDtoDAsync (constant_mem_base + constantMemBytes,
                                         src, mem->size, stream);
                CUDA_CHECK_ABORT (result, "cuMemcpyDtoDAsync");

                constantMemOffsets[i] = constantMemBytes;
                params[i] = constantMemOffsets + i;

                constantMemBytes += mem->size;
              }
            else
              {
                assert (arguments[i].is_raw_ptr == 0);
                if (arguments[i].value)
                  {
                    cl_mem mem = *(void **)arguments[i].value;
                    params[i]
                      = &mem->device_ptrs[device->global_mem_id].mem_ptr;
                  }
                else
                  {
                    params[i] = &null;
                  }
              }
            break;
          }
        case POCL_ARG_TYPE_IMAGE:
        case POCL_ARG_TYPE_SAMPLER:
        case POCL_ARG_TYPE_PIPE:
          POCL_ABORT ("Unhandled argument type for CUDA\n");
          break;
        }
    }

  if (constantMemBytes > constant_mem_size)
    POCL_ABORT ("[CUDA] Total constant buffer size %u exceeds %lu allocated\n",
                constantMemBytes, constant_mem_size);

  unsigned arg_index = meta->num_args;

  if (sharedMemBytes != 0)
    {
      /* Deal with automatic local allocations if there are local function args
       */
      /* TODO: Would be better to remove arguments and make these static GEPs
       */
      for (i = 0; i < meta->num_locals; ++i, ++arg_index)
        {
          size_t size = meta->local_sizes[i];
          size_t align = kdata->alignments[arg_index];
          if (align < 1)
            align = 1;

          /* Pad offset to align memory */
          if (sharedMemBytes % align)
            sharedMemBytes += align - (sharedMemBytes % align);

          sharedMemOffsets[arg_index] = sharedMemBytes;
          sharedMemBytes += size;
          params[arg_index] = sharedMemOffsets + arg_index;
        }
    }

  /* Add global work dimensionality */
  params[arg_index++] = &pc.work_dim;

  /* Add global offsets */
  /* Global offsets are added as arguments in the ptx-gen pass, so we
   * must pass them */
  globalOffsets[0] = pc.global_offset[0];
  globalOffsets[1] = pc.global_offset[1];
  globalOffsets[2] = pc.global_offset[2];
  params[arg_index++] = globalOffsets + 0;
  params[arg_index++] = globalOffsets + 1;
  params[arg_index++] = globalOffsets + 2;

  /* Launch kernel */
  result = cuLaunchKernel (function, pc.num_groups[0], pc.num_groups[1],
                           pc.num_groups[2], pc.local_size[0],
                           pc.local_size[1], pc.local_size[2], sharedMemBytes,
                           stream, params, NULL);
  CUDA_CHECK_ABORT (result, "cuLaunchKernel");
}
