/* CUDA implementation of the OpenCL pow built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

double __nv_pow (double, double);

static float
pocl_cuda_powf (float base, float exponent)
{
  return (float)__nv_pow ((double)base, (double)exponent);
}

DEFINE_NATIVE_F_FF (pow, pocl_cuda_powf (a, b), __nv_pow (a, b))
