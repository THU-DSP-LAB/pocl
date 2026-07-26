/* CUDA implementation of the OpenCL logb built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_logbf (float);
double __nv_logb (double);

#define FLOAT_SIGN_MASK 0x80000000u
#define FLOAT_EXPONENT_MASK 0x7f800000u
#define FLOAT_EXPONENT_SHIFT 23u
#define FLOAT_EXPONENT_BIAS 127
#define FLOAT_SUBNORMAL_EXPONENT_OFFSET 149
#define FLOAT_STORAGE_BITS 32u

static float
pocl_cuda_logbf (float value)
{
  uint magnitude = as_uint (value) & ~FLOAT_SIGN_MASK;
  if (magnitude > FLOAT_EXPONENT_MASK)
    return value;
  if (magnitude == FLOAT_EXPONENT_MASK)
    return INFINITY;
  if (magnitude == 0)
    return -INFINITY;

  uint exponent_bits = magnitude & FLOAT_EXPONENT_MASK;
  if (exponent_bits != 0)
    return (float)((int)(exponent_bits >> FLOAT_EXPONENT_SHIFT)
                   - FLOAT_EXPONENT_BIAS);

  uint leading_bit = FLOAT_STORAGE_BITS - 1u - clz (magnitude);
  return (float)((int)leading_bit - FLOAT_SUBNORMAL_EXPONENT_OFFSET);
}

DEFINE_NATIVE_F_F (logb, pocl_cuda_logbf (a), __nv_logb (a))
