#include "buffer_range_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORACLE_BYTES 16
#define EXHAUSTIVE_BYTES 8
#define MAX_SMALL_RANGES 36

static void
require (int condition, const char *message)
{
  if (!condition)
    {
      fprintf (stderr, "%s\n", message);
      exit (1);
    }
}

static void
expect_gaps (pocl_buffer_range_plan plan, const pocl_buffer_range *expected,
             size_t count)
{
  require (plan.status == CL_SUCCESS, "valid range plan failed");
  require (plan.count == count, "wrong number of gaps");
  for (size_t i = 0; i < count; ++i)
    require (plan.ranges[i].origin == expected[i].origin
                 && plan.ranges[i].size == expected[i].size,
             "wrong gap geometry");
  free (plan.ranges);
}

static void
check_byte_oracle (const pocl_buffer_range *ranges, size_t count,
                   size_t parent_size)
{
  unsigned char expected[ORACLE_BYTES] = { 0 }, actual[ORACLE_BYTES] = { 0 };
  pocl_buffer_range before[3];
  require (parent_size <= ORACLE_BYTES && count <= 3,
           "invalid test dimensions");
  memcpy (before, ranges, count * sizeof (*ranges));
  for (size_t i = 0; i < count; ++i)
    for (size_t j = ranges[i].origin; j < ranges[i].origin + ranges[i].size;
         ++j)
      expected[j] = 1;
  pocl_buffer_range_plan plan
      = pocl_buffer_uncovered_ranges (ranges, count, parent_size);
  require (plan.status == CL_SUCCESS, "small range plan failed");
  require (memcmp (before, ranges, count * sizeof (*ranges)) == 0,
           "planner mutated input");
  for (size_t i = 0; i < plan.count; ++i)
    {
      const pocl_buffer_range gap = plan.ranges[i];
      require (gap.size > 0 && pocl_buffer_range_valid (gap, parent_size),
               "gap escaped parent");
      for (size_t j = gap.origin; j < gap.origin + gap.size; ++j)
        {
          require (!expected[j] && !actual[j],
                   "gap overlaps usage or another gap");
          actual[j] = 1;
        }
    }
  for (size_t i = 0; i < parent_size; ++i)
    require (actual[i] != expected[i], "complement missed a byte");
  free (plan.ranges);
}

static void
exhaustive_unions (void)
{
  pocl_buffer_range ranges[MAX_SMALL_RANGES];
  size_t count = 0;
  for (size_t origin = 0; origin < EXHAUSTIVE_BYTES; ++origin)
    for (size_t size = 1; size <= EXHAUSTIVE_BYTES - origin; ++size)
      ranges[count++] = (pocl_buffer_range){ origin, size };
  require (count == MAX_SMALL_RANGES, "incomplete exhaustive domain");
  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < count; ++j)
      for (size_t k = 0; k < count; ++k)
        {
          const pocl_buffer_range selected[]
              = { ranges[i], ranges[j], ranges[k] };
          check_byte_oracle (selected, 3, EXHAUSTIVE_BYTES);
        }
}

static void
source_regressions (void)
{
  const pocl_buffer_range minimal[] = { { 8, 4 }, { 4, 8 } };
  const pocl_buffer_range minimal_gaps[] = { { 0, 4 }, { 12, 4 } };
  expect_gaps (pocl_buffer_uncovered_ranges (minimal, 2, 16), minimal_gaps, 2);
  const pocl_buffer_range source[] = {
    { 5595136, 161792 },    { 26875904, 3512320 },  { 18892800, 9119744 },
    { 28909568, 1486848 },  { 13811712, 12310528 }, { 32452608, 972800 },
    { 15908864, 11802624 }, { 22702080, 5851136 },  { 4429824, 20142080 },
    { 4190208, 11556864 },  { 31100928, 2449408 },  { 1546240, 15980544 },
    { 31223808, 921600 },   { 6834176, 23021568 },  { 13694976, 1875968 },
    { 22353920, 2742272 }
  };
  const pocl_buffer_range gaps[]
      = { { 0, 1546240 }, { 30396416, 704512 }, { 33550336, 4096 } };
  expect_gaps (pocl_buffer_uncovered_ranges (source, 16, 33554432), gaps, 3);
  const pocl_buffer_range reverse_bridge[]
      = { { 12, 2 }, { 0, 3 }, { 2, 11 } };
  const pocl_buffer_range bridge_gap = { 14, 2 };
  expect_gaps (pocl_buffer_uncovered_ranges (reverse_bridge, 3, 16),
               &bridge_gap, 1);
}

static void
bounds_and_alignment (void)
{
  const pocl_buffer_range invalid = { 256, SIZE_MAX - 127 };
  require (!pocl_buffer_range_valid (invalid, 512), "wrapped end accepted");
  pocl_buffer_range_plan plan
      = pocl_buffer_uncovered_ranges (&invalid, 1, 512);
  require (plan.status == CL_INVALID_VALUE && plan.ranges == NULL,
           "invalid range plan succeeded");
  plan = pocl_buffer_uncovered_ranges (NULL, SIZE_MAX, 0);
  require (plan.status == CL_OUT_OF_HOST_MEMORY, "count overflow accepted");
  plan = pocl_buffer_uncovered_ranges (NULL, 1, 16);
  require (plan.status == CL_INVALID_VALUE, "null input accepted");
  const pocl_buffer_range complete = { 0, SIZE_MAX };
  expect_gaps (pocl_buffer_uncovered_ranges (NULL, 0, SIZE_MAX), &complete, 1);
  const pocl_buffer_range edge = { SIZE_MAX - 7, 7 };
  const pocl_buffer_range prefix = { 0, SIZE_MAX - 7 };
  expect_gaps (pocl_buffer_uncovered_ranges (&edge, 1, SIZE_MAX), &prefix, 1);
  const pocl_buffer_range aligned
      = pocl_buffer_range_align ((pocl_buffer_range){ 7, 10 }, 8);
  require (aligned.origin == 8 && aligned.size == 9, "alignment differs");
  const pocl_buffer_range empty
      = pocl_buffer_range_align ((pocl_buffer_range){ SIZE_MAX - 1, 1 }, 8);
  require (empty.size == 0, "alignment rounding overflowed");
  const pocl_buffer_range last = { 16, 0 };
  const pocl_buffer_range all = { 0, 16 };
  expect_gaps (pocl_buffer_uncovered_ranges (&last, 1, 16), &all, 1);
  const pocl_buffer_range zero_middle = { 8, 0 };
  expect_gaps (pocl_buffer_uncovered_ranges (&zero_middle, 1, 16), &all, 1);
  expect_gaps (pocl_buffer_uncovered_ranges (NULL, 0, 0), NULL, 0);
}

int
main (void)
{
  exhaustive_unions ();
  source_regressions ();
  bounds_and_alignment ();
  puts ("historical range reference + public bounds: 46656 exhaustive triples, original16, bounds "
        "passed");
  return 0;
}
