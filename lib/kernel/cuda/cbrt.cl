/* CUDA implementation of the OpenCL cbrt built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_cbrtf (float);
double __nv_cbrt (double);

DEFINE_NATIVE_F_F (cbrt, __nv_cbrtf (a), __nv_cbrt (a))
