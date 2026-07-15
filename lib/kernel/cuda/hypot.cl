/* CUDA implementation of the OpenCL hypot built-in. */

#include "cuda-templates.h"

float __nv_hypotf (float, float);
double __nv_fabs (double);
double __nv_sqrt (double);

static double
pocl_cuda_hypot (double left, double right)
{
  double left_magnitude = __nv_fabs (left);
  double right_magnitude = __nv_fabs (right);
  if (isinf (left_magnitude) || isinf (right_magnitude))
    return INFINITY;
  if (isnan (left_magnitude) || isnan (right_magnitude))
    return left_magnitude + right_magnitude;

  double maximum
      = left_magnitude > right_magnitude ? left_magnitude : right_magnitude;
  double minimum
      = left_magnitude > right_magnitude ? right_magnitude : left_magnitude;
  if (maximum == 0.0)
    return 0.0;

  double ratio = minimum / maximum;
  return maximum * __nv_sqrt (1.0 + ratio * ratio);
}

DEFINE_NATIVE_F_FF (hypot, __nv_hypotf (a, b), pocl_cuda_hypot (a, b))
