/* Regression test for calling a kernel from another kernel. */

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

static const char KernelSource[] =
    "__kernel void callee(__global int *output, int value)\n"
    "{\n"
    "  output[get_global_id(0)] = value;\n"
    "}\n"
    "__kernel void caller(__global int *output, int value)\n"
    "{\n"
    "  callee(output, value + 1);\n"
    "}\n";

int
main (void)
{
  cl_int err;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;
  cl_program program;
  cl_kernel callee;
  cl_kernel caller;
  cl_mem buffer;
  const char *source = KernelSource;
  const size_t global_size = 1;
  cl_int value = 41;
  cl_int output = 0;

  CHECK_CL_ERROR (
      poclu_get_any_device2 (&context, &device, &queue, &platform));
  program = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));

  callee = clCreateKernel (program, "callee", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel(callee)");
  caller = clCreateKernel (program, "caller", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel(caller)");
  buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (output), NULL,
                           &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

  CHECK_CL_ERROR (clSetKernelArg (caller, 0, sizeof (buffer), &buffer));
  CHECK_CL_ERROR (clSetKernelArg (caller, 1, sizeof (value), &value));
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, caller, 1, NULL, &global_size,
                                          NULL, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0,
                                       sizeof (output), &output, 0, NULL,
                                       NULL));
  TEST_ASSERT (output == 42);

  value = 7;
  CHECK_CL_ERROR (clSetKernelArg (callee, 0, sizeof (buffer), &buffer));
  CHECK_CL_ERROR (clSetKernelArg (callee, 1, sizeof (value), &value));
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, callee, 1, NULL, &global_size,
                                          NULL, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0,
                                       sizeof (output), &output, 0, NULL,
                                       NULL));
  TEST_ASSERT (output == value);

  CHECK_CL_ERROR (clReleaseMemObject (buffer));
  CHECK_CL_ERROR (clReleaseKernel (caller));
  CHECK_CL_ERROR (clReleaseKernel (callee));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  puts ("OK");
  return EXIT_SUCCESS;
}
