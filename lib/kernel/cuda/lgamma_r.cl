/* CUDA implementation of the OpenCL lgamma_r built-in. */

#include "../templates.h"

float __nv_lgammaf (float);
float __nv_sinpif (float);
double __nv_lgamma (double);
double __nv_sinpi (double);

static int
pocl_cuda_gamma_signf (float value)
{
  if (value == 0.0f)
    return signbit (value) ? -1 : 1;
  return value < 0.0f && __nv_sinpif (value) < 0.0f ? -1 : 1;
}

static int
pocl_cuda_gamma_sign (double value)
{
  if (value == 0.0)
    return signbit (value) ? -1 : 1;
  return value < 0.0 && __nv_sinpi (value) < 0.0 ? -1 : 1;
}

static float
pocl_cuda_lgamma_rf (float value, __private int *sign)
{
  *sign = pocl_cuda_gamma_signf (value);
  return __nv_lgammaf (value);
}

static double
pocl_cuda_lgamma_r (double value, __private int *sign)
{
  *sign = pocl_cuda_gamma_sign (value);
  return __nv_lgamma (value);
}

#define __builtin_lgamma_rf pocl_cuda_lgamma_rf
#define __builtin_lgamma_r pocl_cuda_lgamma_r

#undef __IF_FP16
#define __IF_FP16(X)

DEFINE_BUILTIN_V_VPJ (lgamma_r)
