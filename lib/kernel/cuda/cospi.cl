/* CUDA implementation of the OpenCL cospi built-in. */

#include "cuda-templates.h"

float __nv_cospif (float);
double __nv_cospi (double);

DEFINE_NATIVE_F_F (cospi, __nv_cospif (a), __nv_cospi (a))
