/* Regression test for CUDA USE_HOST_PTR registration and rect origins. */

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum
{
  PAGE_BYTES = 4096,
  BUFFER_BYTES = 888,
  SOURCE_OFFSET = 512,
  DESTINATION_OFFSET = 2048,
  RECT_X = 10,
  RECT_WIDTH = 8,
};

int
main (void)
{
  cl_int err;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;
  cl_mem buffer;
  void *allocation = NULL;
  TEST_ASSERT (posix_memalign (&allocation, PAGE_BYTES, PAGE_BYTES) == 0);
  unsigned char *storage = allocation;

  unsigned char *source = storage + SOURCE_OFFSET;
  unsigned char *destination = storage + DESTINATION_OFFSET;
  for (size_t index = 0; index < BUFFER_BYTES; ++index)
    source[index] = (unsigned char)index;

  CHECK_CL_ERROR (
      poclu_get_any_device2 (&context, &device, &queue, &platform));
  buffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                           BUFFER_BYTES, source, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

  memset (destination, 0, BUFFER_BYTES);
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0,
                                       BUFFER_BYTES, destination, 0, NULL,
                                       NULL));
  TEST_ASSERT (memcmp (source, destination, BUFFER_BYTES) == 0);

  const size_t origin[] = { RECT_X, 0, 0 };
  const size_t region[] = { RECT_WIDTH, 1, 1 };
  memset (destination, 0, BUFFER_BYTES);
  CHECK_CL_ERROR (clEnqueueReadBufferRect (
      queue, buffer, CL_TRUE, origin, origin, region, 0, 0, 0, 0,
      destination, 0, NULL, NULL));
  TEST_ASSERT (memcmp (source + RECT_X, destination + RECT_X, RECT_WIDTH) == 0);

  CHECK_CL_ERROR (clReleaseMemObject (buffer));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  free (storage);
  return EXIT_SUCCESS;
}
