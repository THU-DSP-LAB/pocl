/* CUDA implementation of the OpenCL exp10 built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_exp10f (float);
double __nv_exp10 (double);

DEFINE_NATIVE_F_F (exp10, __nv_exp10f (a), __nv_exp10 (a))
