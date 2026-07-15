/* CUDA implementation of the OpenCL cbrt built-in. */

#include "cuda-templates.h"

float __nv_cbrtf (float);
double __nv_cbrt (double);

DEFINE_NATIVE_F_F (cbrt, __nv_cbrtf (a), __nv_cbrt (a))
