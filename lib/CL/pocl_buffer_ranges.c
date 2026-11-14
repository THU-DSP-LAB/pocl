#include "pocl_buffer_ranges.h"

int
pocl_buffer_range_valid (pocl_buffer_range range, size_t parent_size)
{
  return range.origin <= parent_size
         && range.size <= parent_size - range.origin;
}
