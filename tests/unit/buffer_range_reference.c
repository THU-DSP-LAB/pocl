/* Historical gap-planner reference; not linked into the runtime. */
#include "buffer_range_reference.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

pocl_buffer_range
pocl_buffer_range_align (pocl_buffer_range range, size_t alignment)
{
  assert (alignment != 0);
  size_t remainder = range.origin % alignment;
  size_t delta = remainder == 0 ? 0 : alignment - remainder;
  if (delta >= range.size)
    return (pocl_buffer_range){ range.origin, 0 };
  return (pocl_buffer_range){ range.origin + delta, range.size - delta };
}

static int
compare_origins (const void *left, const void *right)
{
  const pocl_buffer_range *a = left;
  const pocl_buffer_range *b = right;
  return (a->origin > b->origin) - (a->origin < b->origin);
}

static cl_int
validate_ranges (const pocl_buffer_range *ranges, size_t count,
                 size_t parent_size)
{
  if (count != 0 && ranges == NULL)
    return CL_INVALID_VALUE;
  for (size_t i = 0; i < count; ++i)
    if (!pocl_buffer_range_valid (ranges[i], parent_size))
      return CL_INVALID_VALUE;
  return CL_SUCCESS;
}

pocl_buffer_range_plan
pocl_buffer_uncovered_ranges (const pocl_buffer_range *ranges, size_t count,
                              size_t parent_size)
{
  pocl_buffer_range_plan plan = { .status = CL_SUCCESS };
  if (count >= SIZE_MAX / sizeof (pocl_buffer_range))
    {
      plan.status = CL_OUT_OF_HOST_MEMORY;
      return plan;
    }
  plan.status = validate_ranges (ranges, count, parent_size);
  if (plan.status != CL_SUCCESS)
    return plan;
  pocl_buffer_range *ordered = malloc ((count + 1) * sizeof (*ordered));
  if (ordered == NULL)
    {
      plan.status = CL_OUT_OF_HOST_MEMORY;
      return plan;
    }
  if (count != 0)
    memcpy (ordered, ranges, count * sizeof (*ordered));
  qsort (ordered, count, sizeof (*ordered), compare_origins);
  size_t position = 0;
  for (size_t i = 0; i < count; ++i)
    {
      const pocl_buffer_range current = ordered[i];
      if (current.size == 0)
        continue;
      if (current.origin > position)
        ordered[plan.count++]
            = (pocl_buffer_range){ position, current.origin - position };
      size_t end = current.origin + current.size;
      if (end > position)
        position = end;
    }
  if (position < parent_size)
    ordered[plan.count++]
        = (pocl_buffer_range){ position, parent_size - position };
  plan.ranges = ordered;
  return plan;
}
