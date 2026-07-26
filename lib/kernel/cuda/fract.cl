/* CUDA implementation of the OpenCL fract built-in. */

#include "../templates.h"

#define FLOAT_FRACT_LIMIT 0x1.fffffep-1f
#define DOUBLE_FRACT_LIMIT 0x1.fffffffffffffp-1
#ifdef cl_khr_fp16
#define HALF_FRACT_LIMIT ((half)0x1.ffcp-1f)
#endif

static float
pocl_cuda_fractf (float value, __private float *integral)
{
  if (isnan (value))
    {
      *integral = value;
      return value;
    }
  if (isinf (value))
    {
      *integral = value;
      return 0.0f;
    }

  float rounded = floor (value);
  *integral = rounded;
  return fmin (value - rounded, FLOAT_FRACT_LIMIT);
}

static double
pocl_cuda_fract (double value, __private double *integral)
{
  if (isnan (value))
    {
      *integral = value;
      return value;
    }
  if (isinf (value))
    {
      *integral = value;
      return 0.0;
    }

  double rounded = floor (value);
  *integral = rounded;
  return fmin (value - rounded, DOUBLE_FRACT_LIMIT);
}

#define __builtin_fractf pocl_cuda_fractf
#define __builtin_fract pocl_cuda_fract

#ifdef cl_khr_fp16
static half
pocl_cuda_fracth (half value, __private float *integral)
{
  if (isnan (value))
    {
      *integral = value;
      return value;
    }
  if (isinf (value))
    {
      *integral = value;
      return (half)0.0f;
    }

  half rounded = floor (value);
  *integral = rounded;
  return fmin (value - rounded, HALF_FRACT_LIMIT);
}

#define __builtin_fractf16 pocl_cuda_fracth
#endif

DEFINE_BUILTIN_V_VPV (fract)
