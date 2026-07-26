/* CUDA implementation of the OpenCL fmod built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_fmodf (float, float);
double __nv_fmod (double, double);

DEFINE_NATIVE_F_FF (fmod, __nv_fmodf (a, b), __nv_fmod (a, b))
