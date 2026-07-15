/* CUDA implementation of the OpenCL fmod built-in. */

#include "cuda-templates.h"

float __nv_fmodf (float, float);
double __nv_fmod (double, double);

DEFINE_NATIVE_F_FF (fmod, __nv_fmodf (a, b), __nv_fmod (a, b))
