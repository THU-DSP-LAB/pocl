/* OpenCL built-in library: subgroup functions

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

uint _pocl_sub_group_active_mask (void);

uint4 _CL_OVERLOADABLE
sub_group_ballot (int predicate)
{
  const uint active = _pocl_sub_group_active_mask ();
  return (uint4)(__nvvm_vote_ballot_sync (active, !!predicate), 0, 0, 0);
}

static uint
lane_bit (void)
{
  return 1u << get_sub_group_local_id ();
}

static uint
sub_group_lane_bits (void)
{
  return UINT_MAX >> (32 - get_max_sub_group_size ());
}

int _CL_OVERLOADABLE
sub_group_ballot_bit_extract (uint4 value, uint index)
{
  return (value.x >> index) & 1u;
}

int _CL_OVERLOADABLE
sub_group_inverse_ballot (uint4 value)
{
  return sub_group_ballot_bit_extract (value, get_sub_group_local_id ());
}

uint _CL_OVERLOADABLE
sub_group_ballot_bit_count (uint4 value)
{
  return popcount (value.x & sub_group_lane_bits ());
}

uint _CL_OVERLOADABLE
sub_group_ballot_inclusive_scan (uint4 value)
{
  const uint current_bit = lane_bit ();
  return popcount (value.x & (current_bit | (current_bit - 1u)));
}

uint _CL_OVERLOADABLE
sub_group_ballot_exclusive_scan (uint4 value)
{
  return popcount (value.x & (lane_bit () - 1u));
}

uint _CL_OVERLOADABLE
sub_group_ballot_find_lsb (uint4 value)
{
  return ctz (value.x & sub_group_lane_bits ());
}

uint _CL_OVERLOADABLE
sub_group_ballot_find_msb (uint4 value)
{
  return 31u - clz (value.x & sub_group_lane_bits ());
}

uint4 _CL_OVERLOADABLE
get_sub_group_eq_mask (void)
{
  return (uint4)(lane_bit (), 0, 0, 0);
}

uint4 _CL_OVERLOADABLE
get_sub_group_lt_mask (void)
{
  return (uint4)(lane_bit () - 1u, 0, 0, 0);
}

uint4 _CL_OVERLOADABLE
get_sub_group_le_mask (void)
{
  const uint current_bit = lane_bit ();
  return (uint4)(current_bit | (current_bit - 1u), 0, 0, 0);
}

uint4 _CL_OVERLOADABLE
get_sub_group_ge_mask (void)
{
  return (uint4)(sub_group_lane_bits () & ~(lane_bit () - 1u), 0, 0, 0);
}

uint4 _CL_OVERLOADABLE
get_sub_group_gt_mask (void)
{
  const uint current_bit = lane_bit ();
  const uint lower_bits = current_bit | (current_bit - 1u);
  return (uint4)(sub_group_lane_bits () & ~lower_bits, 0, 0, 0);
}

/* The CUDA broadcast primitive already selects a lane from the active mask. */
#define NON_UNIFORM_BROADCAST(TYPE)                                          \
  TYPE _CL_OVERLOADABLE sub_group_broadcast (TYPE, uint);                     \
  TYPE _CL_OVERLOADABLE                                                      \
  sub_group_non_uniform_broadcast (TYPE value, uint index)                   \
  {                                                                           \
    return sub_group_broadcast (value, index);                               \
  }

#define NON_UNIFORM_BROADCAST_VECTORS(TYPE)                                  \
  NON_UNIFORM_BROADCAST (TYPE##2)                                             \
  NON_UNIFORM_BROADCAST (TYPE##3)                                             \
  NON_UNIFORM_BROADCAST (TYPE##4)                                             \
  NON_UNIFORM_BROADCAST (TYPE##8)                                             \
  NON_UNIFORM_BROADCAST (TYPE##16)

#define BROADCAST_FIRST(TYPE)                                                \
  TYPE _CL_OVERLOADABLE sub_group_broadcast_first (TYPE value)               \
  {                                                                           \
    return sub_group_non_uniform_broadcast (                                 \
        value, ctz (_pocl_sub_group_active_mask ()));                         \
  }

#define BALLOT_BROADCAST(TYPE)                                                \
  NON_UNIFORM_BROADCAST (TYPE)                                                \
  NON_UNIFORM_BROADCAST_VECTORS (TYPE)                                        \
  BROADCAST_FIRST (TYPE)

BALLOT_BROADCAST (char)
BALLOT_BROADCAST (uchar)
BALLOT_BROADCAST (short)
BALLOT_BROADCAST (ushort)
BALLOT_BROADCAST (int)
BALLOT_BROADCAST (uint)
BALLOT_BROADCAST (long)
BALLOT_BROADCAST (ulong)
BALLOT_BROADCAST (float)
__IF_FP64 (BALLOT_BROADCAST (double))
__IF_FP16 (NON_UNIFORM_BROADCAST_VECTORS (half))
__IF_FP16 (
  half _CL_OVERLOADABLE
  sub_group_non_uniform_broadcast (half value, uint index)
  {
    return as_half (sub_group_non_uniform_broadcast (as_ushort (value), index));
  }
)
__IF_FP16 (BROADCAST_FIRST (half))
