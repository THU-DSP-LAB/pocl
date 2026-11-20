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

enum { POCL_CUDA_MAX_PARTITIONED_LAUNCHES = 8 };

typedef struct pocl_cuda_launch_region_s
{
  unsigned grid[3];
  unsigned block[3];
  unsigned group_offset[3];
} pocl_cuda_launch_region_t;

static unsigned
pocl_cuda_partition_launch (
    const _cl_command_run *run,
    pocl_cuda_launch_region_t regions[POCL_CUDA_MAX_PARTITIONED_LAUNCHES])
{
  memset (regions, 0,
          sizeof (pocl_cuda_launch_region_t)
              * POCL_CUDA_MAX_PARTITIONED_LAUNCHES);
  for (unsigned dimension = 0; dimension < 3; ++dimension)
    {
      regions[0].grid[dimension] = 1;
      regions[0].block[dimension] = 1;
    }

  unsigned region_count = 1;
  for (unsigned dimension = 0; dimension < 3; ++dimension)
    {
      const size_t enqueued = run->pc.local_size[dimension];
      const size_t global = run->global_size[dimension];
      const size_t full_groups = global / enqueued;
      const size_t tail_size = global % enqueued;
      assert (full_groups <= UINT_MAX && tail_size <= UINT_MAX);

      if (full_groups == 0)
        {
          for (unsigned region = 0; region < region_count; ++region)
            regions[region].block[dimension] = tail_size;
          continue;
        }
      for (unsigned region = 0; region < region_count; ++region)
        {
          regions[region].grid[dimension] = full_groups;
          regions[region].block[dimension] = enqueued;
        }
      if (tail_size == 0)
        continue;

      const unsigned old_count = region_count;
      assert (old_count * 2 <= POCL_CUDA_MAX_PARTITIONED_LAUNCHES);
      for (unsigned region = 0; region < old_count; ++region)
        {
          regions[old_count + region] = regions[region];
          regions[old_count + region].grid[dimension] = 1;
          regions[old_count + region].block[dimension] = tail_size;
          regions[old_count + region].group_offset[dimension] = full_groups;
        }
      region_count *= 2;
    }
  return region_count;
}
#include "pocl-cuda-launch-types.h"
#include "pocl-cuda-launch-selection.inc"
#include "pocl-cuda-launch-arguments.inc"
#include "pocl-cuda-launch-dispatch.inc"
