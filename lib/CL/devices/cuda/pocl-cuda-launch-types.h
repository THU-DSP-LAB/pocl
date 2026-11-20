/* Borrowed launch inputs and caller-owned parameter storage. */
#ifndef POCL_CUDA_LAUNCH_TYPES_H
#define POCL_CUDA_LAUNCH_TYPES_H

enum
{
  POCL_CUDA_KERNEL_PARAMETER_BYTES = 4352,
  POCL_CUDA_IMPLICIT_PARAMETER_COUNT = 16
};

typedef struct
{
  CUstream stream;
  _cl_command_node *command;
  cl_device_id device;
  cl_event event;
} pocl_cuda_launch_inputs;

typedef struct
{
  CUfunction function;
  const pocl_cuda_kernel_data_t *kernel_data;
  CUdeviceptr constant_base;
  size_t constant_size;
} pocl_cuda_launch_target;

typedef struct
{
  CUdeviceptr destination;
  CUdeviceptr source;
  size_t bytes;
  CUstream stream;
} pocl_cuda_constant_copy;

typedef struct
{
  const _cl_command_run *run;
  cl_device_id device;
  CUstream stream;
  pocl_cuda_launch_target target;
  void (*copy_constant) (const pocl_cuda_constant_copy *);
  int supplied_table;
} pocl_cuda_argument_context;

typedef struct
{
  void **params;
  unsigned *local_offsets;
  unsigned *constant_offsets;
  size_t abi_count;
  unsigned local_bytes;
  unsigned constant_bytes;
  void *null_pointer;
  cl_uint work_dim;
  unsigned global_offsets[3];
  unsigned global_sizes[3];
  unsigned total_groups[3];
  unsigned enqueued_local_sizes[3];
  unsigned group_offsets[3];
} pocl_cuda_launch_arguments;

#endif
