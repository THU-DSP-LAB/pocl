/* CUDA implementation of the OpenCL powr built-in. */

#include "cuda-templates.h"

double __nv_pow(double, double);

typedef struct {
  double value;
  bool handled;
} pocl_cuda_math_result_t;

static pocl_cuda_math_result_t pocl_cuda_powr_domain_special(double base,
                                                             double exponent) {
  const double not_a_number = as_double((ulong)0x7ff8000000000000UL);
  if (base < 0.0)
    return (pocl_cuda_math_result_t){not_a_number, true};
  if (isnan(base) || isnan(exponent))
    return (pocl_cuda_math_result_t){base + exponent, true};
  if (base == 1.0)
    return (pocl_cuda_math_result_t){isinf(exponent) ? not_a_number : 1.0,
                                     true};
  if (exponent != 0.0)
    return (pocl_cuda_math_result_t){0.0, false};
  if (base == 0.0 || isinf(base))
    return (pocl_cuda_math_result_t){not_a_number, true};
  return (pocl_cuda_math_result_t){1.0, true};
}

static pocl_cuda_math_result_t
pocl_cuda_powr_boundary_special(double base, double exponent) {
  if (base == 0.0)
    return (pocl_cuda_math_result_t){exponent < 0.0 ? INFINITY : 0.0, true};
  if (isinf(base))
    return (pocl_cuda_math_result_t){exponent < 0.0 ? 0.0 : INFINITY, true};
  if (!isinf(exponent))
    return (pocl_cuda_math_result_t){0.0, false};

  const bool below_one = base < 1.0;
  if (exponent < 0.0)
    return (pocl_cuda_math_result_t){below_one ? INFINITY : 0.0, true};
  return (pocl_cuda_math_result_t){below_one ? 0.0 : INFINITY, true};
}

static double pocl_cuda_powr(double base, double exponent) {
  const pocl_cuda_math_result_t domain =
      pocl_cuda_powr_domain_special(base, exponent);
  if (domain.handled)
    return domain.value;

  const pocl_cuda_math_result_t boundary =
      pocl_cuda_powr_boundary_special(base, exponent);
  if (boundary.handled)
    return boundary.value;
  return __nv_pow(base, exponent);
}

static float pocl_cuda_powrf(float base, float exponent) {
  return (float)pocl_cuda_powr((double)base, (double)exponent);
}

DEFINE_NATIVE_F_FF(powr, pocl_cuda_powrf(a, b), pocl_cuda_powr(a, b))
