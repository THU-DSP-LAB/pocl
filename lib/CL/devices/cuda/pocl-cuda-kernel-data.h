/* Per-device compiled CUDA kernel ABI state. */
#ifndef POCL_CUDA_KERNEL_DATA_H
#define POCL_CUDA_KERNEL_DATA_H
typedef struct pocl_cuda_kernel_data_s
{
  CUfunction kernel;
  CUfunction kernel_offsets;
  size_t *alignments;
  /* Exact compiler-map storage count; supplied built-in tables may reserve more. */
  size_t alignment_capacity;
  size_t refcount;
} pocl_cuda_kernel_data_t;

#endif
