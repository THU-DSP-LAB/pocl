/* Regression coverage for non-uniform OpenCL work-groups. */

#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

enum
{
  GLOBAL_SIZE = 10,
  LOCAL_SIZE = 4,
  GLOBAL_OFFSET = 3,
  FIELD_COUNT = 8,
  EXPECTED_GROUP_COUNT = 3,
  TAIL_WORK_ITEM_COUNT = 1,
  TAIL_SUB_GROUP_COUNT = 1,
  SKIP_EXIT_CODE = 77,
};

enum result_field
{
  GLOBAL_ID,
  QUERIED_GLOBAL_SIZE,
  LOCAL_ID,
  QUERIED_LOCAL_SIZE,
  ENQUEUED_LOCAL_SIZE,
  GROUP_ID,
  NUM_GROUPS,
  QUERIED_GLOBAL_OFFSET,
};

static const char *kernel_source
    = "#define FIELD_COUNT 8\n"
      "#define GLOBAL_ID 0\n"
      "#define QUERIED_GLOBAL_SIZE 1\n"
      "#define LOCAL_ID 2\n"
      "#define QUERIED_LOCAL_SIZE 3\n"
      "#define ENQUEUED_LOCAL_SIZE 4\n"
      "#define GROUP_ID 5\n"
      "#define NUM_GROUPS 6\n"
      "#define QUERIED_GLOBAL_OFFSET 7\n"
      "kernel void record_ids(global ulong *results) {"
      "  size_t logical = get_global_id(0) - get_global_offset(0);"
      "  size_t base = logical * FIELD_COUNT;"
      "  results[base + GLOBAL_ID] = get_global_id(0);"
      "  results[base + QUERIED_GLOBAL_SIZE] = get_global_size(0);"
      "  results[base + LOCAL_ID] = get_local_id(0);"
      "  results[base + QUERIED_LOCAL_SIZE] = get_local_size(0);"
      "  results[base + ENQUEUED_LOCAL_SIZE] = get_enqueued_local_size(0);"
      "  results[base + GROUP_ID] = get_group_id(0);"
      "  results[base + NUM_GROUPS] = get_num_groups(0);"
      "  results[base + QUERIED_GLOBAL_OFFSET] = get_global_offset(0);"
      "}";

static const char *subgroup_kernel_source
    = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
      "kernel void record_subgroup_counts(global uint *current,"
      "                                   global uint *enqueued) {"
      "  size_t id = get_global_id(0);"
      "  current[id] = get_num_sub_groups();"
      "  enqueued[id] = get_enqueued_num_sub_groups();"
      "}";

static void
check_results (const cl_ulong results[GLOBAL_SIZE][FIELD_COUNT])
{
  const size_t full_group_items = GLOBAL_SIZE - GLOBAL_SIZE % LOCAL_SIZE;
  for (size_t logical = 0; logical < GLOBAL_SIZE; ++logical)
    {
      const size_t expected_local_size
          = logical < full_group_items ? LOCAL_SIZE : GLOBAL_SIZE % LOCAL_SIZE;
      const size_t expected_local_id = logical % LOCAL_SIZE;
      TEST_ASSERT (results[logical][GLOBAL_ID] == GLOBAL_OFFSET + logical);
      TEST_ASSERT (results[logical][QUERIED_GLOBAL_SIZE] == GLOBAL_SIZE);
      TEST_ASSERT (results[logical][LOCAL_ID] == expected_local_id);
      TEST_ASSERT (results[logical][QUERIED_LOCAL_SIZE] == expected_local_size);
      TEST_ASSERT (results[logical][ENQUEUED_LOCAL_SIZE] == LOCAL_SIZE);
      TEST_ASSERT (results[logical][GROUP_ID] == logical / LOCAL_SIZE);
      TEST_ASSERT (results[logical][NUM_GROUPS] == EXPECTED_GROUP_COUNT);
      TEST_ASSERT (results[logical][QUERIED_GLOBAL_OFFSET] == GLOBAL_OFFSET);
    }
}

static cl_int
run_id_query_test (cl_context context, cl_device_id device,
                   cl_command_queue queue)
{
  cl_int err;
  cl_ulong results[GLOBAL_SIZE][FIELD_COUNT] = { { 0 } };
  cl_mem output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (results),
                                  NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  cl_program program
      = clCreateProgramWithSource (context, 1, &kernel_source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel (program, "record_ids", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (output), &output));

  const cl_ulong zero_pattern = 0;
  CHECK_CL_ERROR (clEnqueueFillBuffer (queue, output, &zero_pattern,
                                      sizeof (zero_pattern), 0,
                                      sizeof (results), 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, NULL, NULL, 0,
                                         NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, output, CL_TRUE, 0,
                                      sizeof (results), results, 0, NULL,
                                      NULL));
  for (size_t index = 0; index < GLOBAL_SIZE * FIELD_COUNT; ++index)
    TEST_ASSERT (((cl_ulong *)results)[index] == 0);

  const size_t global_size = GLOBAL_SIZE;
  const size_t local_size = LOCAL_SIZE;
  const size_t global_offset = GLOBAL_OFFSET;
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (
      queue, kernel, 1, &global_offset, &global_size, &local_size, 0, NULL,
      NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, output, CL_TRUE, 0,
                                      sizeof (results), results, 0, NULL,
                                      NULL));
  check_results (results);

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseMemObject (output));
  return CL_SUCCESS;
}

static cl_int
run_subgroup_query_test (cl_context context, cl_device_id device,
                         cl_command_queue queue)
{
  cl_int err;
  cl_program program = clCreateProgramWithSource (
      context, 1, &subgroup_kernel_source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel
      = clCreateKernel (program, "record_subgroup_counts", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  size_t local_size = 0;
  CHECK_CL_ERROR (clGetKernelWorkGroupInfo (
      kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof (local_size),
      &local_size, NULL));
  const size_t global_size = local_size + TAIL_WORK_ITEM_COUNT;
  size_t enqueued_count = 0;
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
      kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
      sizeof (local_size), &local_size, sizeof (enqueued_count),
      &enqueued_count, NULL));
  TEST_ASSERT (enqueued_count > 1);

  const size_t result_bytes = global_size * sizeof (cl_uint);
  cl_uint *current = calloc (global_size, sizeof (*current));
  cl_uint *enqueued = calloc (global_size, sizeof (*enqueued));
  TEST_ASSERT (current != NULL && enqueued != NULL);
  cl_mem current_buffer = clCreateBuffer (
      context, CL_MEM_WRITE_ONLY, result_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  cl_mem enqueued_buffer = clCreateBuffer (
      context, CL_MEM_WRITE_ONLY, result_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 0, sizeof (current_buffer), &current_buffer));
  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 1, sizeof (enqueued_buffer), &enqueued_buffer));

  CHECK_CL_ERROR (clEnqueueNDRangeKernel (
      queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, current_buffer, CL_TRUE, 0,
                                      result_bytes, current, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, enqueued_buffer, CL_TRUE, 0,
                                      result_bytes, enqueued, 0, NULL, NULL));
  for (size_t id = 0; id < local_size; ++id)
    {
      TEST_ASSERT (current[id] == enqueued_count);
      TEST_ASSERT (enqueued[id] == enqueued_count);
    }
  TEST_ASSERT (current[local_size] == TAIL_SUB_GROUP_COUNT);
  TEST_ASSERT (enqueued[local_size] == enqueued_count);

  free (enqueued);
  free (current);
  CHECK_CL_ERROR (clReleaseMemObject (enqueued_buffer));
  CHECK_CL_ERROR (clReleaseMemObject (current_buffer));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  return CL_SUCCESS;
}

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;
  CHECK_CL_ERROR (
      poclu_get_any_device2 (&context, &device, &queue, &platform));

  cl_bool supported = CL_FALSE;
  CHECK_CL_ERROR (clGetDeviceInfo (device,
                                   CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                                   sizeof (supported), &supported, NULL));
  if (!supported)
    return SKIP_EXIT_CODE;

  CHECK_CL_ERROR (run_id_query_test (context, device, queue));
  if (poclu_supports_extension (device, "cl_khr_subgroups"))
    CHECK_CL_ERROR (run_subgroup_query_test (context, device, queue));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  printf ("OK\n");
  return EXIT_SUCCESS;
}
