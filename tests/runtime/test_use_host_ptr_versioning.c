/* Regression test for versioned USE_HOST_PTR device migrations. */

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

enum
{
  ELEMENT_COUNT = 256,
};

static const char *copy_kernel_source
    = "kernel void copy_values(global const int *source, global int *dest) {"
      "  size_t index = get_global_id(0);"
      "  dest[index] = source[index];"
      "}";

static void
check_values (const cl_int *values, cl_int expected)
{
  for (size_t index = 0; index < ELEMENT_COUNT; ++index)
    TEST_ASSERT (values[index] == expected);
}

int
main (void)
{
  const size_t buffer_size = ELEMENT_COUNT * sizeof (cl_int);
  const size_t global_size = ELEMENT_COUNT;
  const cl_int pattern = 0x12345678;
  cl_int host_source[ELEMENT_COUNT] = { 0 };
  cl_int host_destination[ELEMENT_COUNT] = { 0 };
  cl_int err;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;

  CHECK_CL_ERROR (
      poclu_get_any_device2 (&context, &device, &queue, &platform));

  cl_mem source
      = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                        buffer_size, host_source, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer source");
  cl_mem destination = clCreateBuffer (context, CL_MEM_WRITE_ONLY, buffer_size,
                                       NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer destination");

  cl_program program = clCreateProgramWithSource (
      context, 1, &copy_kernel_source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel (program, "copy_values", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &source));
  CHECK_CL_ERROR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &destination));

  CHECK_CL_ERROR (clEnqueueFillBuffer (queue, source, &pattern,
                                       sizeof (pattern), 0, buffer_size, 0,
                                       NULL, NULL));
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                         NULL, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, destination, CL_TRUE, 0,
                                      buffer_size, host_destination, 0, NULL,
                                      NULL));
  check_values (host_destination, pattern);

  cl_int *mapped = clEnqueueMapBuffer (queue, source, CL_TRUE, CL_MAP_READ, 0,
                                       buffer_size, 0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");
  check_values (mapped, pattern);
  CHECK_CL_ERROR (clEnqueueUnmapMemObject (queue, source, mapped, 0, NULL,
                                          NULL));
  CHECK_CL_ERROR (clFinish (queue));

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseMemObject (destination));
  CHECK_CL_ERROR (clReleaseMemObject (source));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  return EXIT_SUCCESS;
}
