/* Shared state and full byte-oracle helpers for buffer alias regressions. */
#ifndef BUFFER_ALIAS_TEST_HELPERS_H
#define BUFFER_ALIAS_TEST_HELPERS_H
#include <CL/cl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VIEW_COUNT 6
#define PARENT_ALIGNMENT_UNITS 8
#define UNALIGNED_TAIL_BYTES 17

typedef struct
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_command_queue other_queue;
  cl_mem parent;
  cl_mem views[VIEW_COUNT];
  cl_buffer_region regions[VIEW_COUNT];
  cl_program program;
  cl_kernel kernel;
  size_t size;
  size_t alignment;
  unsigned char *expected;
  unsigned char *actual;
} test_state;

static void
require (int condition, const char *operation)
{
  if (!condition)
    {
      fprintf (stderr, "sub-buffer regression failed: %s\n", operation);
      exit (1);
    }
}

static void
check (cl_int status, const char *operation)
{
  if (status != CL_SUCCESS)
    {
      fprintf (stderr, "%s returned OpenCL error %d\n", operation, status);
      exit (1);
    }
}

static test_state
create_state (void)
{
  test_state state = { 0 };
  cl_platform_id platform;
  cl_int error;
  check (clGetPlatformIDs (1, &platform, NULL), "platform");
  check (clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 1, &state.device, NULL),
         "GPU device");
  state.context = clCreateContext (NULL, 1, &state.device, NULL, NULL, &error);
  check (error, "context");
  state.queue = clCreateCommandQueue (state.context, state.device, 0, &error);
  check (error, "queue");
  state.other_queue
      = clCreateCommandQueue (state.context, state.device, 0, &error);
  check (error, "other queue");
  cl_uint alignment_bits;
  check (clGetDeviceInfo (state.device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                          sizeof (alignment_bits), &alignment_bits, NULL),
         "alignment query");
  require (alignment_bits % 8 == 0, "non-byte alignment");
  state.alignment = alignment_bits / 8;
  require (state.alignment > UNALIGNED_TAIL_BYTES,
           "alignment too small for test geometry");
  state.size = PARENT_ALIGNMENT_UNITS * state.alignment + UNALIGNED_TAIL_BYTES;
  state.expected = malloc (state.size);
  state.actual = malloc (state.size);
  require (state.expected != NULL && state.actual != NULL, "host allocations");
  for (size_t i = 0; i < state.size; ++i)
    state.expected[i] = (unsigned char)(i * 29 + 3);
  state.parent = clCreateBuffer (state.context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 state.size, state.expected, &error);
  check (error, "parent");
  return state;
}

static void
check_region (const test_state *state, cl_mem buffer, cl_buffer_region region)
{
  check (clEnqueueReadBuffer (state->other_queue, buffer, CL_TRUE, 0,
                              region.size, state->actual, 0, NULL, NULL),
         "read region");
  for (size_t i = 0; i < region.size; ++i)
    if (state->actual[i] != state->expected[region.origin + i])
      {
        fprintf (stderr,
                 "byte mismatch parent_offset=%zu actual=%02x expected=%02x\n",
                 region.origin + i, state->actual[i],
                 state->expected[region.origin + i]);
        require (0, "full region byte oracle");
      }
}

static void
release_state (const test_state *state)
{
  check (clReleaseKernel (state->kernel), "release kernel");
  check (clReleaseProgram (state->program), "release program");
  for (size_t i = 0; i < VIEW_COUNT; ++i)
    check (clReleaseMemObject (state->views[i]), "release view");
  check (clReleaseMemObject (state->parent), "release parent");
  check (clReleaseCommandQueue (state->queue), "release queue");
  check (clReleaseCommandQueue (state->other_queue), "release other queue");
  check (clReleaseContext (state->context), "release context");
  free (state->actual);
  free (state->expected);
}

#define CONCURRENT_ENQUEUE_ITERATIONS 32

typedef struct
{
  test_state *state;
  atomic_int started;
} concurrent_state;

static void *
concurrent_parent_reads (void *argument)
{
  concurrent_state *shared = argument;
  test_state *state = shared->state;
  unsigned char *bytes = malloc (state->size);
  require (bytes != NULL, "concurrent read storage");
  atomic_store (&shared->started, 1);
  for (unsigned i = 0; i < CONCURRENT_ENQUEUE_ITERATIONS; ++i)
    {
      check (clEnqueueReadBuffer (state->queue, state->parent, CL_TRUE, 0,
                                  state->size, bytes, 0, NULL, NULL),
             "concurrent parent read");
      require (memcmp (bytes, state->expected, state->size) == 0,
               "concurrent full parent oracle");
    }
  free (bytes);
  return NULL;
}

static void
concurrent_enqueue_and_view_lifetime (test_state *state)
{
  concurrent_state shared = { .state = state, .started = 0 };
  pthread_t reader;
  require (pthread_create (&reader, NULL, concurrent_parent_reads, &shared)
               == 0,
           "start concurrent reader");
  while (!atomic_load (&shared.started))
    ;
  for (unsigned i = 0; i < CONCURRENT_ENQUEUE_ITERATIONS; ++i)
    {
      cl_int error;
      cl_buffer_region region = { 3 * state->alignment, 43 + i };
      cl_mem view
          = clCreateSubBuffer (state->parent, CL_MEM_READ_WRITE,
                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
      check (error, "concurrent view creation");
      check (clEnqueueMigrateMemObjects (state->other_queue, 1, &view,
                                         CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                         NULL),
             "concurrent host migration");
      check (clReleaseMemObject (view), "concurrent view release");
    }
  require (pthread_join (reader, NULL) == 0, "join concurrent reader");
  check (clFinish (state->other_queue), "concurrent exports completion");
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
}

typedef struct
{
  test_state *state;
  unsigned char value;
  int export_before_write;
} concurrent_writer;

static void *
same_queue_writer (void *argument)
{
  const concurrent_writer *writer = argument;
  test_state *state = writer->state;
  for (unsigned i = 0; i < CONCURRENT_ENQUEUE_ITERATIONS; ++i)
    {
      if (writer->export_before_write)
        check (clEnqueueMigrateMemObjects (state->queue, 1, &state->views[0],
                                           CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                           NULL),
               "same queue concurrent host");
      check (clEnqueueFillBuffer (state->queue, state->views[0],
                                  &writer->value, 1, 0, state->regions[0].size,
                                  0, NULL, NULL),
             "same queue concurrent writer");
    }
  return NULL;
}

static void
same_queue_concurrent_writers (test_state *state)
{
  cl_int error;
  test_state on_queue = *state;
  on_queue.queue
      = clCreateCommandQueue (state->context, state->device, 0, &error);
  check (error, "cold concurrent request queue");
  concurrent_writer writers[2]
      = { { &on_queue, 0x31, 0 }, { &on_queue, 0xC9, 1 } };
  pthread_t threads[2];
  for (unsigned i = 0; i < 2; ++i)
    require (pthread_create (&threads[i], NULL, same_queue_writer, &writers[i])
                 == 0,
             "start same queue writer");
  for (unsigned i = 0; i < 2; ++i)
    require (pthread_join (threads[i], NULL) == 0, "join same queue writer");
  check (clFinish (on_queue.queue), "same queue writer completion");
  check (clEnqueueReadBuffer (state->other_queue, state->views[0], CL_TRUE, 0,
                              state->regions[0].size, state->actual, 0, NULL,
                              NULL),
         "same queue writer result");
  unsigned char last = state->actual[0];
  require (last == writers[0].value || last == writers[1].value,
           "same queue result is a real writer value");
  for (size_t i = 0; i < state->regions[0].size; ++i)
    require (state->actual[i] == last, "same queue writes remain serialized");
  memset (state->expected + state->regions[0].origin, last,
          state->regions[0].size);
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
  check (clReleaseCommandQueue (on_queue.queue), "release cold request queue");
}

static void
request_queue_lifetime (test_state *state)
{
  cl_int error;
  cl_command_queue queue
      = clCreateCommandQueue (state->context, state->device, 0, &error);
  check (error, "temporary requesting queue");
  const unsigned char value = 0xAC;
  check (clEnqueueFillBuffer (state->queue, state->views[0], &value, 1, 0,
                              state->regions[0].size, 0, NULL, NULL),
         "write before temporary queue export");
  check (clFinish (state->queue), "temporary export writer completion");
  memset (state->expected + state->regions[0].origin, value,
          state->regions[0].size);
  cl_event exported;
  check (clEnqueueMigrateMemObjects (queue, 1, &state->views[0],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                     &exported),
         "temporary queue host export");
  check (clReleaseCommandQueue (queue), "release pending requesting queue");
  check (clWaitForEvents (1, &exported), "request queue released export");
  check (clReleaseEvent (exported), "release temporary export event");
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
}

static void
pending_export_last_references (test_state *state)
{
  cl_int error;
  cl_command_queue queue
      = clCreateCommandQueue (state->context, state->device, 0, &error);
  check (error, "last refs queue");
  cl_mem parent = clCreateBuffer (state->context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  state->size, state->expected, &error);
  check (error, "last refs parent");
  const cl_buffer_region region = { state->alignment, 37 };
  cl_mem view
      = clCreateSubBuffer (parent, CL_MEM_READ_WRITE,
                           CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
  check (error, "last refs view");
  const unsigned char value = 0xDA;
  check (clEnqueueFillBuffer (queue, view, &value, 1, 0, region.size, 0, NULL,
                              NULL),
         "last refs fill");
  check (clEnqueueMigrateMemObjects (
             queue, 1, &view, CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL),
         "last refs host export");
  cl_event read;
  check (clEnqueueReadBuffer (queue, view, CL_FALSE, 0, region.size,
                              state->actual, 0, NULL, &read),
         "last refs read");
  check (clReleaseMemObject (view), "release last user view reference");
  check (clReleaseMemObject (parent), "release last user parent reference");
  check (clReleaseCommandQueue (queue), "release last user queue reference");
  check (clWaitForEvents (1, &read), "last refs read completion");
  for (size_t i = 0; i < region.size; ++i)
    require (state->actual[i] == value, "last refs full byte oracle");
  check (clReleaseEvent (read), "release last refs read event");
}

#endif
