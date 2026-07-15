/* CUDA implementation of the OpenCL frexp built-in. */

#include "../templates.h"

float __nv_frexpf (float, __private int *);
double __nv_frexp (double, __private int *);

#define FLOAT_SIGN_MASK 0x80000000u
#define FLOAT_EXPONENT_MASK 0x7f800000u
#define FLOAT_MANTISSA_MASK 0x007fffffu
#define FLOAT_FREXP_EXPONENT_BITS 0x3f000000u
#define FLOAT_EXPONENT_SHIFT 23u
#define FLOAT_NORMAL_EXPONENT_ADJUST 126
#define FLOAT_SUBNORMAL_EXPONENT_ADJUST 148
#define FLOAT_STORAGE_BITS 32u

static float
pocl_cuda_frexpf (float value, __private int *exponent)
{
  uint bits = as_uint (value);
  uint magnitude = bits & ~FLOAT_SIGN_MASK;
  uint exponent_bits = magnitude & FLOAT_EXPONENT_MASK;
  if (exponent_bits == FLOAT_EXPONENT_MASK || magnitude == 0)
    {
      *exponent = 0;
      return value;
    }
  if (exponent_bits != 0)
    {
      *exponent = (int)(exponent_bits >> FLOAT_EXPONENT_SHIFT)
                  - FLOAT_NORMAL_EXPONENT_ADJUST;
      uint fraction_bits = (bits & FLOAT_SIGN_MASK)
                           | FLOAT_FREXP_EXPONENT_BITS
                           | (bits & FLOAT_MANTISSA_MASK);
      return as_float (fraction_bits);
    }

  uint leading_bit = FLOAT_STORAGE_BITS - 1u - clz (magnitude);
  uint normalized = magnitude << (FLOAT_EXPONENT_SHIFT - leading_bit);
  *exponent = (int)leading_bit - FLOAT_SUBNORMAL_EXPONENT_ADJUST;
  uint fraction_bits = (bits & FLOAT_SIGN_MASK) | FLOAT_FREXP_EXPONENT_BITS
                       | (normalized & FLOAT_MANTISSA_MASK);
  return as_float (fraction_bits);
}

#define __builtin_frexpf pocl_cuda_frexpf
#define __builtin_frexp __nv_frexp

DEFINE_BUILTIN_V_VPJ (frexp)
