/* CUDA implementation of the OpenCL exp10 built-in. */

#include "cuda-templates.h"

float __nv_exp10f (float);
double __nv_exp10 (double);

DEFINE_NATIVE_F_F (exp10, __nv_exp10f (a), __nv_exp10 (a))
