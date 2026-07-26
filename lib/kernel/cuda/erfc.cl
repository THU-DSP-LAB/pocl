/* CUDA implementation of the OpenCL erfc built-in. */

#include "cuda-templates.h"

#ifdef POCL_CORE_MATH_FP16
#undef __IF_FP16
#define __IF_FP16(X)
#endif

float __nv_erfcf(float);
double __nv_erfc(double);

DEFINE_NATIVE_F_F(erfc, __nv_erfcf(a), __nv_erfc(a))
