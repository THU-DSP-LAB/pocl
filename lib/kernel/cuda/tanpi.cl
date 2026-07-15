/* CUDA implementation of the OpenCL tanpi built-in. */

#include "cuda-templates.h"

float __nv_cospif (float);
float __nv_sinpif (float);
double __nv_cospi (double);
double __nv_sinpi (double);

DEFINE_NATIVE_F_F (tanpi, __nv_sinpif (a) / __nv_cospif (a),
                   __nv_sinpi (a) / __nv_cospi (a))
