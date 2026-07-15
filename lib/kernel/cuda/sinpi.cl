/* CUDA implementation of the OpenCL sinpi built-in. */

#include "cuda-templates.h"

float __nv_sinpif (float);
double __nv_sinpi (double);

DEFINE_NATIVE_F_F (sinpi, __nv_sinpif (a), __nv_sinpi (a))
