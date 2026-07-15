/* CUDA implementation of the OpenCL pown built-in. */

#include "../templates.h"

double __nv_pow (double, double);

static float
pocl_cuda_pownf (float value, int exponent)
{
  float magnitude
      = (float)__nv_pow ((double)fabs (value), (double)exponent);
  return (exponent & 1) != 0 ? copysign (magnitude, value) : magnitude;
}

static double
pocl_cuda_pown (double value, int exponent)
{
  double magnitude = __nv_pow (fabs (value), (double)exponent);
  return (exponent & 1) != 0 ? copysign (magnitude, value) : magnitude;
}

#define __builtin_pownf pocl_cuda_pownf
#define __builtin_pown pocl_cuda_pown

#undef __IF_FP16
#define __IF_FP16(X)

DEFINE_BUILTIN_V_VJ (pown)
DEFINE_BUILTIN_V_VI (pown)
