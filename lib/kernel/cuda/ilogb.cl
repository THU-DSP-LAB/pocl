/* CUDA implementation of the OpenCL ilogb built-in. */

#include "../templates.h"

int __nv_ilogbf (float);
int __nv_ilogb (double);

static int
pocl_cuda_ilogbf (float value)
{
  return isnan (value) ? FP_ILOGBNAN : __nv_ilogbf (value);
}

static int
pocl_cuda_ilogb (double value)
{
  return isnan (value) ? FP_ILOGBNAN : __nv_ilogb (value);
}

#define __builtin_ilogbf pocl_cuda_ilogbf
#define __builtin_ilogb pocl_cuda_ilogb

#undef __IF_FP16
#define __IF_FP16(X)

DEFINE_BUILTIN_K_V (ilogb)
