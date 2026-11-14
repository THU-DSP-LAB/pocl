#ifndef BUFFER_RANGE_REFERENCE_H
#define BUFFER_RANGE_REFERENCE_H
#include "pocl_buffer_ranges.h"
  typedef struct
  {
    cl_int status;
    size_t count;
    pocl_buffer_range *ranges;
  } pocl_buffer_range_plan;

  /* Align only a previously checked region with a representable endpoint. */
  pocl_buffer_range pocl_buffer_range_align (pocl_buffer_range range,
                                             size_t alignment);
  pocl_buffer_range_plan
  pocl_buffer_uncovered_ranges (const pocl_buffer_range *ranges, size_t count,
                                size_t parent_size);

#endif
