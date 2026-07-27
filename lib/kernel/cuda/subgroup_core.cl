/* OpenCL built-in library: subgroup core functions

   Copyright (c) 2025 Jan Solanti / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/* Magic variable that is initialized from pocl-ptx-gen.c */
uint _pocl_warp_size;

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_local_linear_id (void);
size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_enqueued_local_size (unsigned int dimindx);
uint _pocl_sub_group_active_mask (void);

void _CL_OVERLOADABLE
sub_group_barrier (cl_mem_fence_flags flags, memory_scope scope)
  __attribute__ ((noduplicate))
{
  sub_group_barrier (flags);
}

uint _CL_OVERLOADABLE
get_sub_group_size (void)
{
  uint max_size = get_max_sub_group_size ();
  size_t first = (get_local_linear_id () / max_size) * max_size;
  size_t remaining = get_local_size (0) * get_local_size (1)
                     * get_local_size (2) - first;
  return remaining < max_size ? (uint)remaining : max_size;
}

uint _CL_OVERLOADABLE
get_max_sub_group_size (void)
{
  return _pocl_warp_size;
}

static uint
sub_group_count_for_size (size_t work_group_size)
{
  const size_t max_sub_group_size = get_max_sub_group_size ();
  return (uint)((work_group_size + max_sub_group_size - 1)
                / max_sub_group_size);
}

uint _CL_OVERLOADABLE
get_num_sub_groups (void)
{
  const size_t work_group_size
      = get_local_size (0) * get_local_size (1) * get_local_size (2);
  return sub_group_count_for_size (work_group_size);
}

uint _CL_OVERLOADABLE
get_enqueued_num_sub_groups (void)
{
  const size_t enqueued_work_group_size
      = get_enqueued_local_size (0) * get_enqueued_local_size (1)
        * get_enqueued_local_size (2);
  return sub_group_count_for_size (enqueued_work_group_size);
}

uint _CL_OVERLOADABLE
get_sub_group_id (void)
{
  return (uint)get_local_linear_id () / get_max_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_sub_group_local_id (void)
{
  return (uint)get_local_linear_id () % get_max_sub_group_size ();
}

static size_t _CL_OVERLOADABLE
get_first_llid (void)
{
  return get_sub_group_id () * get_max_sub_group_size ();
}

#define SUB_GROUP_BROADCAST_T(TYPE)                                           \
  TYPE _CL_OVERLOADABLE sub_group_shuffle (TYPE, uint);                       \
  TYPE _CL_OVERLOADABLE sub_group_broadcast (TYPE val, uint id)               \
  {                                                                           \
    return sub_group_shuffle (val, id);                                       \
  }

SUB_GROUP_BROADCAST_T (int)
SUB_GROUP_BROADCAST_T (uint)
SUB_GROUP_BROADCAST_T (long)
SUB_GROUP_BROADCAST_T (ulong)
__IF_FP16 (SUB_GROUP_BROADCAST_T (half))
SUB_GROUP_BROADCAST_T (float)
__IF_FP64 (SUB_GROUP_BROADCAST_T (double))

#define SUB_GROUP_SCAN_INCLUSIVE_OT(OPNAME, OPERATION, TYPE)                  \
  TYPE _CL_OVERLOADABLE sub_group_scan_inclusive##OPNAME (TYPE val)           \
  {                                                                           \
    uint active_mask = _pocl_sub_group_active_mask ();                        \
    uint own_lane = get_sub_group_local_id ();                                \
    TYPE result = val;                                                        \
    bool initialized = false;                                                 \
    for (uint lane = 0; lane < get_max_sub_group_size (); ++lane)             \
      {                                                                       \
        TYPE b = sub_group_shuffle (val, lane);                               \
        if (lane <= own_lane && (active_mask & (1u << lane)))                 \
          {                                                                   \
            if (!initialized)                                                 \
              {                                                               \
                result = b;                                                   \
                initialized = true;                                           \
              }                                                               \
            else                                                              \
              {                                                               \
                TYPE a = result;                                              \
                result = OPERATION;                                           \
              }                                                               \
          }                                                                   \
      }                                                                       \
    return result;                                                            \
  }

#define SUB_GROUP_SCAN_INCLUSIVE_T(OPNAME, OPERATION)                         \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, int)                        \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, uint)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, long)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, ulong)                      \
  __IF_FP16 (SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, half))           \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, float)                      \
  __IF_FP64 (SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, double))

SUB_GROUP_SCAN_INCLUSIVE_T (_add, (a + b))
SUB_GROUP_SCAN_INCLUSIVE_T (_min, (a > b ? b : a))
SUB_GROUP_SCAN_INCLUSIVE_T (_max, (a > b ? a : b))

#define SUB_GROUP_SCAN_EXCLUSIVE_OT(OPNAME, OPERATION, TYPE, ID)              \
  TYPE _CL_OVERLOADABLE sub_group_scan_exclusive##OPNAME (TYPE val)           \
  {                                                                           \
    uint active_mask = _pocl_sub_group_active_mask ();                        \
    uint own_lane = get_sub_group_local_id ();                                \
    TYPE result = ID;                                                         \
    for (uint lane = 0; lane < get_max_sub_group_size (); ++lane)             \
      {                                                                       \
        TYPE b = sub_group_shuffle (val, lane);                               \
        if (lane < own_lane && (active_mask & (1u << lane)))                  \
          {                                                                   \
            TYPE a = result;                                                  \
            result = OPERATION;                                               \
          }                                                                   \
      }                                                                       \
    return result;                                                            \
  }

SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, int, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, long, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, ulong, 0)
__IF_FP16 (SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, half, (half)0.0f))
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, float, 0.0f)
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, double, 0.0))

SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, int, INT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, uint, UINT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, long, LONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, ulong, ULONG_MAX)
__IF_FP16 (
  SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, half, (half)(+INFINITY)))
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, float, +INFINITY)
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (
  _min, a > b ? b : a, double, (double)(+INFINITY)))

SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, int, INT_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, long, LONG_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, ulong, 0)
__IF_FP16 (
  SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, half, (half)(-INFINITY)))
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, float, -INFINITY)
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (
  _max, a > b ? a : b, double, (double)(-INFINITY)))
