/* OpenCL built-in library: extended bit-ops extension

   Copyright (c) 2026 Michal Babej / Tampere University

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

#include "templates.h"

#define IMPLEMENT_BITFIELD_INSERT(GTYPE, SGTYPE, UGTYPE, BITS)                \
  GTYPE __attribute__ ((overloadable)) bitfield_insert (                      \
    GTYPE base, GTYPE insert, uint offset, uint count)                        \
  {                                                                           \
    if (count == 0)                                                           \
      return base;                                                            \
    union { GTYPE value; UGTYPE bits; } base_bits = { .value = base };         \
    union { GTYPE value; UGTYPE bits; } insert_bits = { .value = insert };     \
    UGTYPE field_mask = count == BITS                                         \
                             ? (UGTYPE)(-1)                                   \
                             : (((UGTYPE)1 << count) - (UGTYPE)1);             \
    UGTYPE shifted_mask = field_mask << offset;                               \
    union { GTYPE value; UGTYPE bits; } result = {                            \
      .bits = (base_bits.bits & ~shifted_mask)                                \
              | ((insert_bits.bits & field_mask) << offset)                   \
    };                                                                         \
    return result.value;                                                      \
  }

IMPLEMENT_BITFIELD_INSERT (char, char, uchar, 8)
IMPLEMENT_BITFIELD_INSERT (char2, char, uchar2, 8)
IMPLEMENT_BITFIELD_INSERT (char3, char, uchar3, 8)
IMPLEMENT_BITFIELD_INSERT (char4, char, uchar4, 8)
IMPLEMENT_BITFIELD_INSERT (char8, char, uchar8, 8)
IMPLEMENT_BITFIELD_INSERT (char16, char, uchar16, 8)
IMPLEMENT_BITFIELD_INSERT (uchar, uchar, uchar, 8)
IMPLEMENT_BITFIELD_INSERT (uchar2, uchar, uchar2, 8)
IMPLEMENT_BITFIELD_INSERT (uchar3, uchar, uchar3, 8)
IMPLEMENT_BITFIELD_INSERT (uchar4, uchar, uchar4, 8)
IMPLEMENT_BITFIELD_INSERT (uchar8, uchar, uchar8, 8)
IMPLEMENT_BITFIELD_INSERT (uchar16, uchar, uchar16, 8)
IMPLEMENT_BITFIELD_INSERT (short, short, ushort, 16)
IMPLEMENT_BITFIELD_INSERT (short2, short, ushort2, 16)
IMPLEMENT_BITFIELD_INSERT (short3, short, ushort3, 16)
IMPLEMENT_BITFIELD_INSERT (short4, short, ushort4, 16)
IMPLEMENT_BITFIELD_INSERT (short8, short, ushort8, 16)
IMPLEMENT_BITFIELD_INSERT (short16, short, ushort16, 16)
IMPLEMENT_BITFIELD_INSERT (ushort, ushort, ushort, 16)
IMPLEMENT_BITFIELD_INSERT (ushort2, ushort, ushort2, 16)
IMPLEMENT_BITFIELD_INSERT (ushort3, ushort, ushort3, 16)
IMPLEMENT_BITFIELD_INSERT (ushort4, ushort, ushort4, 16)
IMPLEMENT_BITFIELD_INSERT (ushort8, ushort, ushort8, 16)
IMPLEMENT_BITFIELD_INSERT (ushort16, ushort, ushort16, 16)
IMPLEMENT_BITFIELD_INSERT (int, int, uint, 32)
IMPLEMENT_BITFIELD_INSERT (int2, int, uint2, 32)
IMPLEMENT_BITFIELD_INSERT (int3, int, uint3, 32)
IMPLEMENT_BITFIELD_INSERT (int4, int, uint4, 32)
IMPLEMENT_BITFIELD_INSERT (int8, int, uint8, 32)
IMPLEMENT_BITFIELD_INSERT (int16, int, uint16, 32)
IMPLEMENT_BITFIELD_INSERT (uint, uint, uint, 32)
IMPLEMENT_BITFIELD_INSERT (uint2, uint, uint2, 32)
IMPLEMENT_BITFIELD_INSERT (uint3, uint, uint3, 32)
IMPLEMENT_BITFIELD_INSERT (uint4, uint, uint4, 32)
IMPLEMENT_BITFIELD_INSERT (uint8, uint, uint8, 32)
IMPLEMENT_BITFIELD_INSERT (uint16, uint, uint16, 32)

__IF_INT64 (
IMPLEMENT_BITFIELD_INSERT (long, long, ulong, 64)
IMPLEMENT_BITFIELD_INSERT (long2, long, ulong2, 64)
IMPLEMENT_BITFIELD_INSERT (long3, long, ulong3, 64)
IMPLEMENT_BITFIELD_INSERT (long4, long, ulong4, 64)
IMPLEMENT_BITFIELD_INSERT (long8, long, ulong8, 64)
IMPLEMENT_BITFIELD_INSERT (long16, long, ulong16, 64)
IMPLEMENT_BITFIELD_INSERT (ulong, ulong, ulong, 64)
IMPLEMENT_BITFIELD_INSERT (ulong2, ulong, ulong2, 64)
IMPLEMENT_BITFIELD_INSERT (ulong3, ulong, ulong3, 64)
IMPLEMENT_BITFIELD_INSERT (ulong4, ulong, ulong4, 64)
IMPLEMENT_BITFIELD_INSERT (ulong8, ulong, ulong8, 64)
IMPLEMENT_BITFIELD_INSERT (ulong16, ulong, ulong16, 64))

#define IMPLEMENT_BITFIELD_EXTRACT_SIGNED(GTYPE, ASTYPE, UGTYPE, BITS)        \
  GTYPE __attribute__ ((overloadable)) bitfield_extract_signed (              \
    GTYPE base, uint offset, uint count)                                      \
  {                                                                           \
    if (count == 0)                                                           \
      return (GTYPE)0;                                                        \
    union { GTYPE value; UGTYPE bits; } source = { .value = base };            \
    UGTYPE field_mask = count == BITS                                         \
                             ? (UGTYPE)(-1)                                   \
                             : (((UGTYPE)1 << count) - (UGTYPE)1);             \
    UGTYPE value = (source.bits >> offset) & field_mask;                      \
    UGTYPE sign_bit = (UGTYPE)1 << (count - 1);                               \
    union { GTYPE value; UGTYPE bits; } result = {                            \
      .bits = (value ^ sign_bit) - sign_bit                                   \
    };                                                                         \
    return result.value;                                                      \
  }                                                                           \
  GTYPE __attribute__ ((overloadable)) bitfield_extract_signed (              \
    UGTYPE base, uint offset, uint count)                                     \
  {                                                                           \
    union { GTYPE value; UGTYPE bits; } source = { .bits = base };             \
    return bitfield_extract_signed (source.value, offset, count);             \
  }

IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char, as_char, uchar, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char2, as_char2, uchar2, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char3, as_char3, uchar3, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char4, as_char4, uchar4, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char8, as_char8, uchar8, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (char16, as_char16, uchar16, 8)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short, as_short, ushort, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short2, as_short2, ushort2, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short3, as_short3, ushort3, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short4, as_short4, ushort4, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short8, as_short8, ushort8, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (short16, as_short16, ushort16, 16)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int, as_int, uint, 32)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int2, as_int2, uint2, 32)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int3, as_int3, uint3, 32)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int4, as_int4, uint4, 32)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int8, as_int8, uint8, 32)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (int16, as_int16, uint16, 32)
__IF_INT64 (
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long, as_long, ulong, 64)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long2, as_long2, ulong2, 64)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long3, as_long3, ulong3, 64)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long4, as_long4, ulong4, 64)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long8, as_long8, ulong8, 64)
IMPLEMENT_BITFIELD_EXTRACT_SIGNED (long16, as_long16, ulong16, 64))

#define IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED(GTYPE, ASTYPE, UGTYPE, BITS)      \
  UGTYPE __attribute__ ((overloadable)) bitfield_extract_unsigned (           \
    UGTYPE base, uint offset, uint count)                                     \
  {                                                                           \
    if (count == 0)                                                           \
      return (UGTYPE)0;                                                       \
    UGTYPE field_mask = count == BITS                                         \
                             ? (UGTYPE)(-1)                                   \
                             : (((UGTYPE)1 << count) - (UGTYPE)1);             \
    return (base >> offset) & field_mask;                                     \
  }                                                                           \
  UGTYPE __attribute__ ((overloadable)) bitfield_extract_unsigned (           \
    GTYPE base, uint offset, uint count)                                      \
  {                                                                           \
    union { GTYPE value; UGTYPE bits; } source = { .value = base };            \
    return bitfield_extract_unsigned (source.bits, offset, count);            \
  }

IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char, as_uchar, uchar, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char2, as_uchar2, uchar2, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char3, as_uchar3, uchar3, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char4, as_uchar4, uchar4, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char8, as_uchar8, uchar8, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (char16, as_uchar16, uchar16, 8)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short, as_ushort, ushort, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short2, as_ushort2, ushort2, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short3, as_ushort3, ushort3, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short4, as_ushort4, ushort4, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short8, as_ushort8, ushort8, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (short16, as_ushort16, ushort16, 16)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int, as_uint, uint, 32)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int2, as_uint2, uint2, 32)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int3, as_uint3, uint3, 32)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int4, as_uint4, uint4, 32)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int8, as_uint8, uint8, 32)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (int16, as_uint16, uint16, 32)
__IF_INT64 (
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long, as_ulong, ulong, 64)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long2, as_ulong2, ulong2, 64)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long3, as_ulong3, ulong3, 64)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long4, as_ulong4, ulong4, 64)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long8, as_ulong8, ulong8, 64)
IMPLEMENT_BITFIELD_EXTRACT_UNSIGNED (long16, as_ulong16, ulong16, 64))

#define POCL_BIT_REVERSE_FALLBACK                                             \
  ({                                                                          \
    union { gtype value; ugtype bits; } source = { .value = a };              \
    ugtype input = source.bits;                                               \
    ugtype reversed = (ugtype)0;                                              \
    for (uint bit = 0; bit < CHAR_BIT * sizeof (sgtype); ++bit)               \
      {                                                                        \
        reversed = (reversed << 1) | (input & (ugtype)1);                     \
        input >>= 1;                                                          \
      }                                                                        \
    union { gtype value; ugtype bits; } result = { .bits = reversed };         \
    result.value;                                                             \
  })
DEFINE_EXPR_G_G (bit_reverse, POCL_BIT_REVERSE_FALLBACK)
#undef POCL_BIT_REVERSE_FALLBACK
