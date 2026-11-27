/* Copyright (c) 2026 pocl developers
 * SPDX-License-Identifier: MIT
 */

/* Each work-item completes its share synchronously. The event is reusable
   across copies. The existing wait_group_events barrier joins those shares. */
#define DEFINE_EXTENDED_COPY(DST_AS, SRC_AS)                                   \
  __attribute__ ((overloadable))                                             \
  event_t async_work_group_copy_3D3D (                                       \
      DST_AS void *dst, size_t dst_offset, const SRC_AS void *src,             \
      size_t src_offset, size_t element_bytes, size_t width, size_t lines,     \
      size_t planes, size_t src_line, size_t src_plane, size_t dst_line,       \
      size_t dst_plane, event_t event)                                        \
  {                                                                         \
    size_t id = get_local_linear_id ();                                      \
    size_t group_size = get_local_size (0) * get_local_size (1)               \
                        * get_local_size (2);                                \
    size_t row_bytes = width * element_bytes;                                \
    DST_AS uchar *dst_base = (DST_AS uchar *)dst + dst_offset * element_bytes;\
    const SRC_AS uchar *src_base                                             \
        = (const SRC_AS uchar *)src + src_offset * element_bytes;             \
    for (size_t plane = 0; plane < planes; ++plane)                          \
      for (size_t line = 0; line < lines; ++line)                            \
        {                                                                   \
          size_t dst_row = (plane * dst_plane + line * dst_line)             \
                           * element_bytes;                                 \
          size_t src_row = (plane * src_plane + line * src_line)             \
                           * element_bytes;                                 \
          for (size_t byte = id; byte < row_bytes; byte += group_size)       \
            dst_base[dst_row + byte] = src_base[src_row + byte];             \
        }                                                                   \
    return event;                                                           \
  }                                                                         \
                                                                            \
  __attribute__ ((overloadable))                                             \
  event_t async_work_group_copy_2D2D (                                       \
      DST_AS void *dst, size_t dst_offset, const SRC_AS void *src,             \
      size_t src_offset, size_t element_bytes, size_t width, size_t lines,     \
      size_t src_line, size_t dst_line, event_t event)                        \
  {                                                                         \
    return async_work_group_copy_3D3D (                                      \
        dst, dst_offset, src, src_offset, element_bytes, width, lines, 1,     \
        src_line, 0, dst_line, 0, event);                                    \
  }

DEFINE_EXTENDED_COPY (__local, __global)
DEFINE_EXTENDED_COPY (__global, __local)
