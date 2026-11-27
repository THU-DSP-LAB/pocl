/* Masked subgroup arithmetic for CUDA targets with synchronized shuffles. */

uint _pocl_sub_group_active_mask (void);

#define SUB_GROUP_FOLD(NAME, TYPE, CALCULATE, FIRST_VALUE, IDENTITY)        \
  TYPE _CL_OVERLOADABLE sub_group_shuffle (TYPE, uint);                     \
  static TYPE _CL_OVERLOADABLE                                               \
  fold_##NAME (TYPE value, uint first, uint end)                             \
  {                                                                           \
    const uint active = _pocl_sub_group_active_mask ();                     \
    TYPE result = IDENTITY;                                                   \
    bool initialized = false;                                                 \
    for (uint lane = 0; lane < get_max_sub_group_size (); ++lane)             \
      {                                                                       \
        const TYPE b = sub_group_shuffle (value, lane);                      \
        if (lane >= first && lane < end && (active & (1u << lane)))          \
          {                                                                   \
            const TYPE a = result;                                            \
            result = initialized ? (CALCULATE) : (FIRST_VALUE);              \
            initialized = true;                                               \
          }                                                                   \
      }                                                                       \
    return result;                                                            \
  }                                                                           \
  TYPE _CL_OVERLOADABLE sub_group_non_uniform_reduce_##NAME (TYPE value)   \
  {                                                                           \
    return fold_##NAME (value, 0, get_max_sub_group_size ());                 \
  }                                                                           \
  TYPE _CL_OVERLOADABLE                                                      \
  sub_group_non_uniform_scan_inclusive_##NAME (TYPE value)                  \
  {                                                                           \
    return fold_##NAME (value, 0, get_sub_group_local_id () + 1u);            \
  }                                                                           \
  TYPE _CL_OVERLOADABLE                                                      \
  sub_group_non_uniform_scan_exclusive_##NAME (TYPE value)                  \
  {                                                                           \
    return fold_##NAME (value, 0, get_sub_group_local_id ());                 \
  }                                                                           \
  TYPE _CL_OVERLOADABLE                                                      \
  sub_group_clustered_reduce_##NAME (TYPE value, uint cluster_size)         \
  {                                                                           \
    const uint first = get_sub_group_local_id () & ~(cluster_size - 1u);     \
    return fold_##NAME (value, first, first + cluster_size);                 \
  }

#define SUB_GROUP_INTEGER(TYPE, ARITH_TYPE, LOW, HIGH)                      \
  SUB_GROUP_FOLD (add, TYPE, (TYPE)((ARITH_TYPE)a + (ARITH_TYPE)b), b, 0)   \
  SUB_GROUP_FOLD (mul, TYPE, (TYPE)((ARITH_TYPE)a * (ARITH_TYPE)b), b, 1)   \
  SUB_GROUP_FOLD (min, TYPE, min (a, b), b, HIGH)                             \
  SUB_GROUP_FOLD (max, TYPE, max (a, b), b, LOW)                              \
  SUB_GROUP_FOLD (and, TYPE, a & b, b, (TYPE)~(TYPE)0)                       \
  SUB_GROUP_FOLD (or, TYPE, a | b, b, 0)                                     \
  SUB_GROUP_FOLD (xor, TYPE, a ^ b, b, 0)

SUB_GROUP_INTEGER (char, uint, CHAR_MIN, CHAR_MAX)
SUB_GROUP_INTEGER (uchar, uint, 0, UCHAR_MAX)
SUB_GROUP_INTEGER (short, uint, SHRT_MIN, SHRT_MAX)
SUB_GROUP_INTEGER (ushort, uint, 0, USHRT_MAX)
SUB_GROUP_INTEGER (int, uint, INT_MIN, INT_MAX)
SUB_GROUP_INTEGER (uint, uint, 0, UINT_MAX)
SUB_GROUP_INTEGER (long, ulong, LONG_MIN, LONG_MAX)
SUB_GROUP_INTEGER (ulong, ulong, 0, ULONG_MAX)

#define SUB_GROUP_FLOAT(TYPE)                                                \
  SUB_GROUP_FOLD (add, TYPE, a + b, b, (TYPE)0)                               \
  SUB_GROUP_FOLD (mul, TYPE, a * b, b, (TYPE)1)                               \
  SUB_GROUP_FOLD (min, TYPE, fmin (a, b), b, (TYPE)INFINITY)                  \
  SUB_GROUP_FOLD (max, TYPE, fmax (a, b), b, (TYPE)(-INFINITY))

SUB_GROUP_FLOAT (float)
__IF_FP16 (SUB_GROUP_FLOAT (half))
__IF_FP64 (SUB_GROUP_FLOAT (double))

SUB_GROUP_FOLD (logical_and, int, (!!a && !!b), !!b, 1)
SUB_GROUP_FOLD (logical_or, int, (!!a || !!b), !!b, 0)
SUB_GROUP_FOLD (logical_xor, int, (!!a != !!b), !!b, 0)
