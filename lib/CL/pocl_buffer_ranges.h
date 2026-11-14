/* Overflow-safe buffer region validation. */
#ifndef POCL_BUFFER_RANGES_H
#define POCL_BUFFER_RANGES_H

#include <CL/cl.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    size_t origin;
    size_t size;
  } pocl_buffer_range;

  int pocl_buffer_range_valid (pocl_buffer_range range, size_t parent_size);

#ifdef __cplusplus
}
#endif
#endif
