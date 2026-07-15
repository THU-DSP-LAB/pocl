/* Regression test for a constant argument removed as unused by LLVM. */

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

static const char KernelSource[] =
    "__kernel void unused_constant(__constant int *unused, "
    "                              __global int *output)\n"
    "{\n"
    "  output[get_global_id(0)] = 42;\n"
    "}\n";

enum
{
  UNUSED_ELEMENT_COUNT = 128,
  EXPECTED_OUTPUT = 42,
};

int
main (void)
{
  cl_int err;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;
  cl_program program;
  cl_kernel kernel;
  cl_mem unused_buffer;
  cl_mem output_buffer;
  const char *source = KernelSource;
  const size_t global_size = 1;
  cl_int unused[UNUSED_ELEMENT_COUNT] = { 0 };
  cl_int output = 0;

  CHECK_CL_ERROR (
      poclu_get_any_device2 (&context, &device, &queue, &platform));
  program = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  kernel = clCreateKernel (program, "unused_constant", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  unused_buffer = clCreateBuffer (context, CL_MEM_COPY_HOST_PTR,
                                  sizeof (unused), unused, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer(unused)");
  output_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (output),
                                  NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer(output)");

  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 0, sizeof (unused_buffer), &unused_buffer));
  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 1, sizeof (output_buffer), &output_buffer));
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                          NULL, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, output_buffer, CL_TRUE, 0,
                                       sizeof (output), &output, 0, NULL,
                                       NULL));
  TEST_ASSERT (output == EXPECTED_OUTPUT);

  CHECK_CL_ERROR (clReleaseMemObject (output_buffer));
  CHECK_CL_ERROR (clReleaseMemObject (unused_buffer));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  puts ("OK");
  return EXIT_SUCCESS;
}
