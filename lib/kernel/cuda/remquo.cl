/* CUDA implementation of the OpenCL remquo built-in. */

#include "../templates.h"

#define POCL_REMQUO_QUOTIENT_MASK 0x7fu

typedef struct {
  double remainder;
  uint quotient;
} pocl_cuda_reduction_t;

typedef struct {
  double remainder;
  int quotient;
  bool handled;
} pocl_cuda_remquo_result_t;

static pocl_cuda_remquo_result_t pocl_cuda_remquo_special(double dividend,
                                                          double divisor) {
  const double not_a_number = as_double((ulong)0x7ff8000000000000UL);
  if (isnan(dividend) || isnan(divisor) || isinf(dividend) || divisor == 0.0)
    return (pocl_cuda_remquo_result_t){not_a_number, 0, true};
  if (isinf(divisor) || dividend == 0.0)
    return (pocl_cuda_remquo_result_t){dividend, 0, true};
  if (fabs(dividend) != fabs(divisor))
    return (pocl_cuda_remquo_result_t){0.0, 0, false};

  const int quotient = dividend == divisor ? 1 : -1;
  return (pocl_cuda_remquo_result_t){copysign(0.0, dividend), quotient, true};
}

static pocl_cuda_reduction_t
pocl_cuda_divide_magnitudes(double dividend, double divisor, int shifts) {
  pocl_cuda_reduction_t result = {dividend, 0u};
  for (int index = shifts; index > 0; --index) {
    result.quotient <<= 1;
    if (result.remainder >= divisor) {
      result.remainder -= divisor;
      result.quotient += 1u;
    }
    result.remainder += result.remainder;
  }

  result.quotient <<= 1;
  if (result.remainder > divisor) {
    result.remainder -= divisor;
    result.quotient += 1u;
  }
  return result;
}

static pocl_cuda_reduction_t
pocl_cuda_round_quotient(pocl_cuda_reduction_t reduction, double divisor) {
  const double twice_remainder = 2.0 * reduction.remainder;
  const bool above_half = divisor < twice_remainder;
  const bool odd_tie =
      divisor == twice_remainder && (reduction.quotient & 1u) != 0u;
  if (!above_half && !odd_tie)
    return reduction;

  reduction.remainder -= divisor;
  reduction.quotient += 1u;
  return reduction;
}

static pocl_cuda_reduction_t pocl_cuda_reduce_magnitudes(double dividend,
                                                         double divisor) {
  const int dividend_exponent = ilogb(dividend);
  const int divisor_exponent = ilogb(divisor);
  const int exponent_delta = dividend_exponent - divisor_exponent;
  if (exponent_delta < -1)
    return pocl_cuda_round_quotient((pocl_cuda_reduction_t){dividend, 0u},
                                    divisor);

  const double scaled_divisor = ldexp(divisor, -divisor_exponent);
  const double scaled_dividend = ldexp(dividend, -dividend_exponent);
  pocl_cuda_reduction_t reduction =
      exponent_delta == -1
          ? (pocl_cuda_reduction_t){ldexp(scaled_dividend, -1), 0u}
          : pocl_cuda_divide_magnitudes(scaled_dividend, scaled_divisor,
                                        exponent_delta);
  reduction = pocl_cuda_round_quotient(reduction, scaled_divisor);
  reduction.remainder = ldexp(reduction.remainder, divisor_exponent);
  return reduction;
}

static double pocl_cuda_remquo(double dividend, double divisor, int *quotient) {
  const pocl_cuda_remquo_result_t special =
      pocl_cuda_remquo_special(dividend, divisor);
  if (special.handled) {
    *quotient = special.quotient;
    return special.remainder;
  }

  const pocl_cuda_reduction_t reduction =
      pocl_cuda_reduce_magnitudes(fabs(dividend), fabs(divisor));
  const int quotient_sign = signbit(dividend) == signbit(divisor) ? 1 : -1;
  *quotient =
      quotient_sign * (int)(reduction.quotient & POCL_REMQUO_QUOTIENT_MASK);
  return signbit(dividend) ? -reduction.remainder : reduction.remainder;
}

static float pocl_cuda_remquof(float dividend, float divisor, int *quotient) {
  return (float)pocl_cuda_remquo((double)dividend, (double)divisor, quotient);
}

#define __builtin_remquof pocl_cuda_remquof
#define __builtin_remquo pocl_cuda_remquo

#undef __IF_FP16
#define __IF_FP16(X)

DEFINE_BUILTIN_V_VVPJ(remquo)
