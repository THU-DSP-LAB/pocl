/* pocl-cuda.c - driver for CUDA devices

   Copyright (c) 2016-2017 James Price / University of Bristol
                 2024 Henry Linjamäki / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "config.h"

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl-cuda-queue.h"
#include "pocl-cuda.h"
#include "pocl-ptx-gen.h"
#include "pocl.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "spirv_queries.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#define CUDNN_CALL(f)                                                         \
  {                                                                           \
    cudnnStatus_t err = (f);                                                  \
    if (err != CUDNN_STATUS_SUCCESS)                                          \
      {                                                                       \
        POCL_ABORT ("  CUDNN Error occurred: %d", err);                       \
      }                                                                       \
  }

cudnnHandle_t cudnn;
#endif // ENABLE_CUDNN

#define CUDA_CALL(f)                                                          \
  {                                                                           \
    cudaError_t err = (f);                                                    \
    if (err != cudaSuccess)                                                   \
      {                                                                       \
        POCL_ABORT ("  Error occurred: %d", err);                             \
      }                                                                       \
  }

#define CUDA_BUILTIN_KERNELS 6
#define CUDA_LATEST_CONFORMANCE_VERSION ""
#define CUDA_BINARY_FORMAT_VERSION "9"
static const char *cuda_builtin_kernels[CUDA_BUILTIN_KERNELS]
    = { "pocl.mul.i32",
        "pocl.add.i32",
        "pocl.dnn.conv2d_int8_relu",
        "pocl.sgemm.local.f32",
        "pocl.sgemm.tensor.f16f16f32",
        "pocl.sgemm_ab.tensor.f16f16f32" };

#define OPENCL_BUILTIN_KERNELS 5
static const char *opencl_builtin_kernels[OPENCL_BUILTIN_KERNELS] = {
  "pocl.abs.f32",
  // from common builtin kernels:
  "pocl.add.i8",
  "org.khronos.openvx.scale_image.nn.u8",
  "org.khronos.openvx.scale_image.bl.u8",
  "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32",
};

#ifdef ENABLE_CUDNN
#define CUDNN_BUILTIN_KERNELS 1
static const char *cudnn_builtin_kernels[CUDNN_BUILTIN_KERNELS]
    = { "pocl.dnn.conv2d.nchw.f32" };
#else
#define CUDNN_BUILTIN_KERNELS 0
#endif

/* The frexp functions are wrappers required because the __nv_frexp and the
 * LLVM's frexp intrinsics have different signatures. */
static const char *cuda_native_device_aux_funcs[] =
  {"frexpf_f32_i32", "frexp_f64_i32", NULL};


static const cl_ulong CUDA_NANOSECONDS_PER_MILLISECOND = 1000000;

enum
{
  CUDA_HOST_REGISTRATION_NOT_OWNED = 0,
  CUDA_HOST_REGISTRATION_OWNED = 1
};

typedef struct
{
  cl_int status;
  void *device_pointer;
  void *registration_base;
  uint64_t owns_registration;
} pocl_cuda_host_registration_t;

void pocl_cuda_svm_copy_async (CUstream, void *restrict, const void *restrict,
                               size_t);

typedef struct pocl_cuda_kernel_data_s
{
  CUfunction kernel;
  CUfunction kernel_offsets;
  size_t *alignments;
  size_t refcount;
} pocl_cuda_kernel_data_t;

typedef struct pocl_cuda_program_data_s
{
  CUmodule module;
  CUmodule module_offsets;
  CUdeviceptr constant_mem_base;
  CUdeviceptr constant_mem_base_offsets;
  size_t constant_mem_size;
  size_t constant_mem_size_offsets;
  void *align_map;
  void *align_map_offsets;
} pocl_cuda_program_data_t;

typedef struct pocl_cuda_event_data_s
{
  CUevent start;
  CUevent end;
  volatile int events_ready;
  pthread_cond_t event_cond;
  unsigned num_ext_events;
  int use_threads;
} pocl_cuda_event_data_t;

typedef struct pocl_cuda_device_data_s
{
  CUdevice device;
  CUcontext context;
  CUevent epoch_event;
  cl_ulong epoch;
  char libdevice[PATH_MAX];
  pocl_lock_t compile_lock;
  int supports_cu_mem_host_register;
  int supports_managed_memory;
  unsigned sm;  /* Targeted SM version. Represented as MAJOR * 10 + MINOR. */
  unsigned ptx; /* Targeted PTX version. Represented as MAJOR * 10 + MINOR. */
  int warp_size;
  cl_bool available;

  pocl_cuda_kernel_data_t cuda_builtin_kernels_data[CUDA_BUILTIN_KERNELS];
  pocl_cuda_kernel_data_t cudnn_builtin_kernels_data[CUDNN_BUILTIN_KERNELS];
  pocl_cuda_program_data_t cuda_builtin_kernels_program;
  pocl_cuda_program_data_t cudnn_builtin_kernels_program;
  int cuda_builtin_kernels_built;

} pocl_cuda_device_data_t;

extern uint64_t pocl_num_devices;


/* Keep the legacy CUDA driver in one translation unit while separating
   implementation responsibilities into reviewable source fragments. */
#include "pocl-cuda-target.inc"
#include "pocl-cuda-device.inc"
#include "pocl-cuda-memory.inc"
#include "pocl-cuda-transfer.inc"
#include "pocl-cuda-program.inc"
#include "pocl-cuda-kernel.inc"
#include "pocl-cuda-submit.inc"
#include "pocl-cuda-event.inc"
#include "pocl-cuda-svm.inc"
