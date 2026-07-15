#ifndef POCL_GEOMETRIC_HELPERS_H
#define POCL_GEOMETRIC_HELPERS_H

#define DEFINE_MAX_ABS(TYPE)                                                \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE a)                        \
  {                                                                         \
    return fabs(a);                                                         \
  }                                                                         \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE##2 a)                     \
  {                                                                         \
    return fmax(fabs(a.s0), fabs(a.s1));                                    \
  }                                                                         \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE##3 a)                     \
  {                                                                         \
    return fmax(fmax(fabs(a.s0), fabs(a.s1)), fabs(a.s2));                  \
  }                                                                         \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE##4 a)                     \
  {                                                                         \
    return fmax(pocl_max_abs(a.lo), pocl_max_abs(a.hi));                    \
  }                                                                         \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE##8 a)                     \
  {                                                                         \
    return fmax(pocl_max_abs(a.lo), pocl_max_abs(a.hi));                    \
  }                                                                         \
  static TYPE _CL_OVERLOADABLE pocl_max_abs(TYPE##16 a)                    \
  {                                                                         \
    return fmax(pocl_max_abs(a.lo), pocl_max_abs(a.hi));                    \
  }

#ifdef cl_khr_fp16
DEFINE_MAX_ABS(half)
#endif
DEFINE_MAX_ABS(float)
#ifdef cl_khr_fp64
DEFINE_MAX_ABS(double)
#endif

#undef DEFINE_MAX_ABS

#endif
