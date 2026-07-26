/* CUDA implementation of the OpenCL cospi built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_cospif (float);
double __nv_cospi (double);

DEFINE_NATIVE_F_F (cospi, __nv_cospif (a), __nv_cospi (a))
