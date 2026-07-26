/* CUDA implementation of the OpenCL sinpi built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_sinpif (float);
double __nv_sinpi (double);

DEFINE_NATIVE_F_F (sinpi, __nv_sinpif (a), __nv_sinpi (a))
