/* CUDA implementation of the OpenCL lgamma built-in. */

#include "cuda-templates.h"

float __nv_lgammaf (float);
double __nv_lgamma (double);

DEFINE_NATIVE_F_F (lgamma, __nv_lgammaf (a), __nv_lgamma (a))
