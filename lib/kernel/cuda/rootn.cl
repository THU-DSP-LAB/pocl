/* CUDA implementation of the OpenCL rootn built-in. */

#include "../templates.h"

double __nv_exp2(double);
double __nv_ldexp(double, int);
double __nv_log2(double);
double __nv_sqrt(double);
double __nv_cbrt(double);

#define POCL_DOUBLE_FRACTION_MASK ((ulong)0x000fffffffffffffUL)
#define POCL_DOUBLE_BIASED_EXPONENT_MASK 0x7ff
#define POCL_DOUBLE_EXPONENT_SHIFT 52
#define POCL_DOUBLE_EXPONENT_BIAS 1023
#define POCL_DOUBLE_SUBNORMAL_EXPONENT_OFFSET 1074
#define POCL_ULONG_HIGHEST_BIT 63

static double pocl_cuda_rootn_zero(double value, int exponent) {
  const bool odd_exponent = (exponent & 1) != 0;
  if (exponent > 0)
    return odd_exponent ? copysign(0.0, value) : 0.0;
  return odd_exponent ? copysign((double)INFINITY, value) : INFINITY;
}

static double pocl_cuda_normalize(double value, int *binary_exponent) {
  const ulong bits = as_ulong(value);
  const ulong fraction = bits & POCL_DOUBLE_FRACTION_MASK;
  const int biased_exponent = (int)(bits >> POCL_DOUBLE_EXPONENT_SHIFT) &
                              POCL_DOUBLE_BIASED_EXPONENT_MASK;
  if (biased_exponent != 0) {
    const ulong normalized_bits = fraction | ((ulong)POCL_DOUBLE_EXPONENT_BIAS
                                              << POCL_DOUBLE_EXPONENT_SHIFT);
    *binary_exponent = biased_exponent - POCL_DOUBLE_EXPONENT_BIAS;
    return as_double(normalized_bits);
  }

  const int highest_fraction_bit = POCL_ULONG_HIGHEST_BIT - (int)clz(fraction);
  const int shift = POCL_DOUBLE_EXPONENT_SHIFT - highest_fraction_bit;
  const ulong normalized_bits =
      ((fraction << shift) & POCL_DOUBLE_FRACTION_MASK) |
      ((ulong)POCL_DOUBLE_EXPONENT_BIAS << POCL_DOUBLE_EXPONENT_SHIFT);
  *binary_exponent =
      highest_fraction_bit - POCL_DOUBLE_SUBNORMAL_EXPONENT_OFFSET;
  return as_double(normalized_bits);
}

static double pocl_cuda_root_magnitude(double value, uint degree,
                                       bool reciprocal) {
  if (isnan(value) || isinf(value))
    return reciprocal ? 1.0 / value : value;
  if (degree == 1u)
    return reciprocal ? 1.0 / value : value;
  if (degree == 2u)
    return reciprocal ? 1.0 / __nv_sqrt(value) : __nv_sqrt(value);
  if (degree == 3u)
    return reciprocal ? 1.0 / __nv_cbrt(value) : __nv_cbrt(value);

  int binary_exponent;
  const double mantissa = pocl_cuda_normalize(value, &binary_exponent);
  const long signed_degree = (long)degree;
  long exponent_quotient = (long)binary_exponent / signed_degree;
  const long exponent_remainder =
      (long)binary_exponent - exponent_quotient * signed_degree;
  double fractional_exponent =
      ((double)exponent_remainder + __nv_log2(mantissa)) / (double)degree;
  if (reciprocal) {
    exponent_quotient = -exponent_quotient;
    fractional_exponent = -fractional_exponent;
  }
  return __nv_ldexp(__nv_exp2(fractional_exponent),
                    (int)exponent_quotient);
}

static double pocl_cuda_rootn(double value, int exponent) {
  const double not_a_number = as_double((ulong)0x7ff8000000000000UL);
  if (exponent == 0)
    return not_a_number;
  if (value == 0.0)
    return pocl_cuda_rootn_zero(value, exponent);

  const uint degree = exponent < 0 ? (uint)(-(long)exponent) : (uint)exponent;
  if (value < 0.0 && (degree & 1u) == 0u)
    return not_a_number;

  const double magnitude =
      pocl_cuda_root_magnitude(fabs(value), degree, exponent < 0);
  return copysign(magnitude, value);
}

static float pocl_cuda_rootnf(float value, int exponent) {
  return (float)pocl_cuda_rootn((double)value, exponent);
}

#define __builtin_rootnf pocl_cuda_rootnf
#define __builtin_rootn pocl_cuda_rootn

#undef __IF_FP16
#define __IF_FP16(X)

DEFINE_BUILTIN_V_VJ(rootn)
DEFINE_BUILTIN_V_VI(rootn)
