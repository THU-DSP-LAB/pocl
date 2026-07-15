/* Verify that process exit drains device workers even when the application
 * leaves OpenCL objects alive. */

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

enum {
  BUFFER_BYTES = 1024 * 1024,
  COMMAND_COUNT = 128,
};

typedef struct exit_test_objects_s {
  cl_int status;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_mem buffer;
} exit_test_objects_t;

static exit_test_objects_t create_test_objects(void) {
  cl_int err;
  exit_test_objects_t objects = { 0 };
  cl_platform_id platform;
  objects.status = poclu_get_any_device2(&objects.context, &objects.device,
                                         &objects.queue, &platform);
  if (objects.status != CL_SUCCESS)
    return objects;

  objects.buffer = clCreateBuffer(objects.context, CL_MEM_READ_WRITE,
                                  BUFFER_BYTES, NULL, &err);
  objects.status = err;
  return objects;
}

static cl_int enqueue_fills(const exit_test_objects_t *objects) {
  for (cl_uint pattern = 0; pattern < COMMAND_COUNT; ++pattern) {
    cl_int status = clEnqueueFillBuffer(
        objects->queue, objects->buffer, &pattern, sizeof(pattern), 0,
        BUFFER_BYTES, 0, NULL, NULL);
    if (status != CL_SUCCESS)
      return status;
  }
  return CL_SUCCESS;
}

int main(void) {
  exit_test_objects_t live_objects = create_test_objects();
  CHECK_CL_ERROR(live_objects.status);
  CHECK_CL_ERROR(enqueue_fills(&live_objects));

  exit_test_objects_t event_owned_objects = create_test_objects();
  CHECK_CL_ERROR(event_owned_objects.status);
  CHECK_CL_ERROR(enqueue_fills(&event_owned_objects));
  CHECK_CL_ERROR(clReleaseMemObject(event_owned_objects.buffer));
  CHECK_CL_ERROR(clReleaseCommandQueue(event_owned_objects.queue));
  CHECK_CL_ERROR(clReleaseContext(event_owned_objects.context));

  /* Leave one object graph alive, while the other is retained only by queued
   * events. Process-exit cleanup must drain both ownership states. */
  puts("OK");
  return EXIT_SUCCESS;
}
