#include "buffer_alias_test_helpers.h"

static const char dirty_source[]
    = "__kernel void dirty(__global uchar *p, uchar tag) {"
      "size_t i=get_global_id(0); p[i]=(uchar)(p[i]^tag); }";

static void
build_dirty_kernel (test_state *state)
{
  const char *source = dirty_source;
  cl_int error;
  state->program
      = clCreateProgramWithSource (state->context, 1, &source, NULL, &error);
  check (error, "program");
  error = clBuildProgram (state->program, 1, &state->device, NULL, NULL, NULL);
  if (error != CL_SUCCESS)
    {
      size_t size = 0;
      check (clGetProgramBuildInfo (state->program, state->device,
                                    CL_PROGRAM_BUILD_LOG, 0, NULL, &size),
             "build log size");
      char *log = calloc (size + 1, 1);
      require (log != NULL, "build log allocation");
      check (clGetProgramBuildInfo (state->program, state->device,
                                    CL_PROGRAM_BUILD_LOG, size, log, NULL),
             "build log");
      fwrite (log, 1, size, stderr);
      free (log);
    }
  check (error, "kernel build");
  state->kernel = clCreateKernel (state->program, "dirty", &error);
  check (error, "kernel");
}

static void
create_views (test_state *state)
{
  const size_t a = state->alignment;
  const cl_buffer_region regions[VIEW_COUNT]
      = { { 2 * a, a + 13 },     { a, 2 * a + 29 }, { 4 * a, a + 7 },
          { 3 * a, 2 * a + 41 }, { 0, a - 13 },     { 7 * a, a + 17 } };
  memcpy (state->regions, regions, sizeof (regions));
  for (size_t i = 0; i < VIEW_COUNT; ++i)
    {
      cl_int error;
      state->views[i] = clCreateSubBuffer (state->parent, CL_MEM_READ_WRITE,
                                           CL_BUFFER_CREATE_TYPE_REGION,
                                           &regions[i], &error);
      check (error, "overlapping view");
    }
}

static void
dirty_views (test_state *state)
{
  const cl_buffer_region parent_region = { 0, state->size };
  for (size_t i = 0; i < VIEW_COUNT; ++i)
    {
      const unsigned char tag = (unsigned char)(0x81 + i * 7);
      const cl_buffer_region region = state->regions[i];
      check (
          clSetKernelArg (state->kernel, 0, sizeof (cl_mem), &state->views[i]),
          "set buffer");
      check (clSetKernelArg (state->kernel, 1, sizeof (tag), &tag), "set tag");
      check (clEnqueueNDRangeKernel (state->queue, state->kernel, 1, NULL,
                                     &region.size, NULL, 0, NULL, NULL),
             "dirty launch");
      check (clFinish (state->queue), "dirty completion");
      for (size_t byte = region.origin; byte < region.origin + region.size;
           ++byte)
        state->expected[byte] ^= tag;
      check_region (state, state->parent, parent_region);
    }
}

static void
parent_and_copy_versions (test_state *state)
{
  for (size_t i = 0; i < state->size; ++i)
    state->expected[i] = (unsigned char)(i * 19 + 0xA3);
  check (clEnqueueWriteBuffer (state->queue, state->parent, CL_TRUE, 0,
                               state->size, state->expected, 0, NULL, NULL),
         "update parent");
  for (size_t i = 0; i < VIEW_COUNT; ++i)
    check_region (state, state->views[i], state->regions[i]);
  const size_t bytes = state->regions[0].size;
  check (clEnqueueCopyBuffer (state->queue, state->views[0], state->views[5],
                              0, 0, bytes, 0, NULL, NULL),
         "copy aliases");
  check (clFinish (state->queue), "copy completion");
  memmove (state->expected + state->regions[5].origin,
           state->expected + state->regions[0].origin, bytes);
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
}

static void
reject_invalid_region (const test_state *state, cl_buffer_region region,
                       cl_int expected)
{
  cl_int status;
  cl_mem result = clCreateSubBuffer (
      state->parent, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
  if (result != NULL)
    check (clReleaseMemObject (result), "release invalid region");
  require (result == NULL && status == expected,
           "overflow/bounds API rejection");
}

static void
bounds_api (const test_state *state)
{
  const size_t a = state->alignment;
  reject_invalid_region (state, (cl_buffer_region){ a, SIZE_MAX - a / 2 },
                         CL_INVALID_VALUE);
  reject_invalid_region (state, (cl_buffer_region){ SIZE_MAX - a + 1, 2 * a },
                         CL_INVALID_VALUE);
  reject_invalid_region (state, (cl_buffer_region){ state->size, 1 },
                         CL_INVALID_VALUE);
  reject_invalid_region (state, (cl_buffer_region){ 0, 0 },
                         CL_INVALID_BUFFER_SIZE);
  reject_invalid_region (state, (cl_buffer_region){ 1, a },
                         CL_MISALIGNED_SUB_BUFFER_OFFSET);
  cl_int error;
  cl_buffer_region nested = { 0, 1 };
  cl_mem result = clCreateSubBuffer (
      state->views[0], 0, CL_BUFFER_CREATE_TYPE_REGION, &nested, &error);
  require (result == NULL && error == CL_INVALID_MEM_OBJECT,
           "sub-buffer parent rejection");
}

static void
flag_contract (const test_state *state)
{
  const cl_mem_flags parent_flags[]
      = { CL_MEM_READ_ONLY,
          CL_MEM_WRITE_ONLY,
          CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
          CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
          CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
          CL_MEM_READ_WRITE };
  const cl_mem_flags child_flags[] = {
    CL_MEM_READ_WRITE,      CL_MEM_READ_ONLY,      CL_MEM_HOST_READ_ONLY,
    CL_MEM_HOST_WRITE_ONLY, CL_MEM_HOST_READ_ONLY, CL_MEM_COPY_HOST_PTR
  };
  const cl_buffer_region region = { 0, state->alignment };
  for (size_t i = 0; i < sizeof (parent_flags) / sizeof (*parent_flags); ++i)
    {
      cl_int error;
      cl_mem parent = clCreateBuffer (state->context, parent_flags[i],
                                      state->size, NULL, &error);
      check (error, "flag parent");
      cl_mem child
          = clCreateSubBuffer (parent, child_flags[i],
                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
      require (child == NULL && error == CL_INVALID_VALUE,
               "parent/child access flag rejection");
      child = clCreateSubBuffer (parent, 0, CL_BUFFER_CREATE_TYPE_REGION,
                                 &region, &error);
      check (error, "inherited access flags");
      check (clReleaseMemObject (child), "release flag child");
      check (clReleaseMemObject (parent), "release flag parent");
    }
}

static void
partial_write_and_map (test_state *state)
{
  const unsigned char value = 0x57;
  const size_t offset = 3, bytes = 29;
  check (clEnqueueFillBuffer (state->queue, state->views[0], &value, 1, offset,
                              bytes, 0, NULL, NULL),
         "partial fill");
  check (clFinish (state->queue), "partial fill completion");
  memset (state->expected + state->regions[0].origin + offset, value, bytes);
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
  const cl_map_flags modes[]
      = { CL_MAP_READ, CL_MAP_WRITE, CL_MAP_READ | CL_MAP_WRITE };
  for (size_t i = 0; i < sizeof (modes) / sizeof (*modes); ++i)
    {
      cl_int error;
      unsigned char *mapped = clEnqueueMapBuffer (
          state->queue, state->views[3], CL_TRUE, modes[i], offset, bytes, 0,
          NULL, NULL, &error);
      check (error, "partial view map");
      if (modes[i] & CL_MAP_READ)
        require (memcmp (mapped,
                         state->expected + state->regions[3].origin + offset,
                         bytes)
                     == 0,
                 "mapped original content");
      if (modes[i] & CL_MAP_WRITE)
        {
          memset (mapped, 0x39 + i, bytes);
          memset (state->expected + state->regions[3].origin + offset,
                  0x39 + i, bytes);
        }
      check (clEnqueueUnmapMemObject (state->queue, state->views[3], mapped, 0,
                                      NULL, NULL),
             "partial view unmap");
      check (clFinish (state->queue), "unmap completion");
      check_region (state, state->parent,
                    (cl_buffer_region){ 0, state->size });
    }
  check (clEnqueueWriteBuffer (state->queue, state->parent, CL_TRUE, offset,
                               bytes, state->expected + offset, 0, NULL, NULL),
         "partial parent write");
  check_region (state, state->views[4], state->regions[4]);
}

static void
undefined_view_migration (test_state *state)
{
  const cl_mem_migration_flags modes[]
      = { CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
          CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
              | CL_MIGRATE_MEM_OBJECT_HOST };
  for (size_t i = 0; i < sizeof (modes) / sizeof (*modes); ++i)
    {
      cl_mem view = state->views[0];
      unsigned char value = (unsigned char)(0x63 + i);
      check (clEnqueueMigrateMemObjects (state->queue, 1, &view, modes[i], 0,
                                         NULL, NULL),
             "undefined view");
      check (clEnqueueFillBuffer (state->queue, view, &value, 1, 0,
                                  state->regions[0].size, 0, NULL, NULL),
             "define view after migration");
      check (clFinish (state->queue), "undefined migration completion");
      memset (state->expected + state->regions[0].origin, value,
              state->regions[0].size);
      check_region (state, state->parent,
                    (cl_buffer_region){ 0, state->size });
    }
}

static void
disjoint_queue_progress (test_state *state)
{
  cl_int error;
  cl_event gate = clCreateUserEvent (state->context, &error), done;
  check (error, "progress gate");
  const unsigned char value = 0x74;
  check (clEnqueueFillBuffer (state->queue, state->views[0], &value, 1, 0,
                              state->regions[0].size, 1, &gate, NULL),
         "gated disjoint write");
  check (clFlush (state->queue), "flush gated write");
  check (clEnqueueMigrateMemObjects (state->queue, 1, &state->views[0],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                     NULL),
         "queued gated host export before independent export");
  check (clEnqueueMigrateMemObjects (state->other_queue, 1, &state->views[5],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                     &done),
         "independent host migration");
  check (clFlush (state->other_queue), "flush independent migration");
  check (clWaitForEvents (1, &done), "independent migration progress");
  check (clSetUserEventStatus (gate, CL_COMPLETE), "release progress gate");
  check (clFinish (state->queue), "gated write completion");
  memset (state->expected + state->regions[0].origin, value,
          state->regions[0].size);
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
  check (clReleaseEvent (done), "release migration event");
  check (clReleaseEvent (gate), "release progress gate event");
}

static void
pending_writer_new_view (test_state *state)
{
  cl_int error;
  cl_command_queue fresh_queue
      = clCreateCommandQueue (state->context, state->device, 0, &error);
  check (error, "pending export split queue");
  cl_event gate = clCreateUserEvent (state->context, &error), written;
  check (error, "new view gate");
  const unsigned char value = 0x91;
  check (clEnqueueFillBuffer (state->queue, state->views[3], &value, 1, 0,
                              state->regions[3].size, 1, &gate, &written),
         "pending view writer");
  check (clEnqueueMigrateMemObjects (state->other_queue, 1, &state->views[3],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 1, &written,
                                     NULL),
         "original pending host export");
  const cl_buffer_region region = { 3 * state->alignment, 19 };
  cl_mem view
      = clCreateSubBuffer (state->parent, CL_MEM_READ_WRITE,
                           CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
  check (error, "view created after pending writer");
  check (clEnqueueMigrateMemObjects (fresh_queue, 1, &view,
                                     CL_MIGRATE_MEM_OBJECT_HOST, 1, &written,
                                     NULL),
         "pending split host migration");
  check (clSetUserEventStatus (gate, CL_COMPLETE), "release new view gate");
  check (clFinish (state->queue), "pending writer completion");
  check (clFinish (fresh_queue), "split pending export on new queue");
  check (clFinish (state->other_queue), "pending split migration completion");
  memset (state->expected + state->regions[3].origin, value,
          state->regions[3].size);
  check_region (state, view, region);
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
  check (clReleaseMemObject (view), "release new view");
  check (clReleaseCommandQueue (fresh_queue), "release split export queue");
  check (clReleaseEvent (written), "release pending writer event");
  check (clReleaseEvent (gate), "release new view gate event");
}

static void
queued_view_lifetime (test_state *state)
{
  cl_int error;
  const cl_buffer_region region = { state->alignment, state->alignment + 13 };
  cl_mem parent = clCreateBuffer (state->context, CL_MEM_READ_WRITE,
                                  state->size, NULL, &error);
  check (error, "lifetime parent");
  cl_mem view
      = clCreateSubBuffer (parent, CL_MEM_READ_WRITE,
                           CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
  check (error, "lifetime view");
  const unsigned char value = 0xA7;
  cl_event written;
  check (clEnqueueFillBuffer (state->queue, view, &value, 1, 0, region.size, 0,
                              NULL, &written),
         "lifetime fill");
  check (clEnqueueReadBuffer (state->other_queue, view, CL_FALSE, 0,
                              region.size, state->actual, 1, &written, NULL),
         "lifetime queued read");
  check (clReleaseMemObject (view), "release queued view");
  check (clReleaseMemObject (parent), "release queued parent");
  check (clFinish (state->other_queue), "lifetime completion");
  check (clFinish (state->queue), "lifetime writer completion");
  for (size_t i = 0; i < region.size; ++i)
    require (state->actual[i] == value, "queued view lifetime byte oracle");
  check (clReleaseEvent (written), "release lifetime event");
}

static void
multiple_pending_writers (test_state *state)
{
  cl_int error;
  cl_event gate = clCreateUserEvent (state->context, &error), written[2];
  check (error, "multiple writer gate");
  const unsigned char value = 0xBC;
  const size_t indices[] = { 0, 5 };
  cl_mem views[] = { state->views[0], state->views[5] };
  cl_command_queue queues[] = { state->queue, state->other_queue };
  for (unsigned i = 0; i < 2; ++i)
    check (clEnqueueFillBuffer (queues[i], views[i], &value, 1, 0,
                                state->regions[indices[i]].size, 1, &gate,
                                &written[i]),
           "multiple pending writers");
  check (clEnqueueMigrateMemObjects (state->other_queue, 2, views,
                                     CL_MIGRATE_MEM_OBJECT_HOST, 2, written,
                                     NULL),
         "multiple pending host exports");
  check (clSetUserEventStatus (gate, CL_COMPLETE), "release multiple gate");
  check (clFinish (state->queue), "multiple first writer completion");
  check (clFinish (state->other_queue), "multiple export completion");
  for (unsigned i = 0; i < 2; ++i)
    {
      memset (state->expected + state->regions[indices[i]].origin, value,
              state->regions[indices[i]].size);
      check (clReleaseEvent (written[i]), "release multiple writer event");
    }
  check_region (state, state->parent, (cl_buffer_region){ 0, state->size });
  check (clReleaseEvent (gate), "release multiple gate event");
}

static void
failed_pending_dependency (test_state *state)
{
  cl_int error, execution;
  cl_event gate = clCreateUserEvent (state->context, &error), written,
           migrated;
  check (error, "failure gate");
  const unsigned char value = 0xE1;
  check (clEnqueueFillBuffer (state->queue, state->views[3], &value, 1, 0,
                              state->regions[3].size, 1, &gate, &written),
         "failed pending fill");
  check (clEnqueueMigrateMemObjects (state->other_queue, 1, &state->views[3],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 1, &written,
                                     &migrated),
         "failure propagation migration");
  check (clSetUserEventStatus (gate, -1), "fail dependency gate");
  require (clWaitForEvents (1, &migrated)
               == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
           "failed dependency host waiter returns error");
  check (clGetEventInfo (migrated, CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof (execution), &execution, NULL),
         "failed migration status");
  require (execution < CL_COMPLETE, "failure propagated to migration event");
  check (clFinish (state->queue), "failed writer drained");
  check (clFinish (state->other_queue), "failed migration drained");
  check (clReleaseEvent (written), "release failed writer");
  check (clReleaseEvent (migrated), "release failed migration");
  check (clReleaseEvent (gate), "release failed gate");
}

static void
independent_read_progress (test_state *state)
{
  cl_int error;
  cl_event gate = clCreateUserEvent (state->context, &error), done;
  check (error, "read progress gate");
  unsigned char *bytes = malloc (state->regions[0].size);
  require (bytes != NULL, "read progress storage");
  check (clEnqueueReadBuffer (state->queue, state->views[0], CL_FALSE, 0,
                              state->regions[0].size, bytes, 1, &gate, NULL),
         "gated readonly view");
  check (clEnqueueMigrateMemObjects (state->other_queue, 1, &state->views[0],
                                     CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL,
                                     &done),
         "independent readonly host export");
  check (clWaitForEvents (1, &done), "read-read independent progress");
  check (clSetUserEventStatus (gate, CL_COMPLETE), "release readonly gate");
  check (clFinish (state->queue), "readonly completion");
  require (memcmp (bytes, state->expected + state->regions[0].origin,
                   state->regions[0].size)
               == 0,
           "readonly data oracle");
  check (clReleaseEvent (done), "release readonly migration");
  check (clReleaseEvent (gate), "release readonly gate event");
  free (bytes);
}

int
main (void)
{
  test_state state = create_state ();
  create_views (&state);
  bounds_api (&state);
  flag_contract (&state);
  build_dirty_kernel (&state);
  dirty_views (&state);
  parent_and_copy_versions (&state);
  partial_write_and_map (&state);
  undefined_view_migration (&state);
  independent_read_progress (&state);
  disjoint_queue_progress (&state);
  pending_writer_new_view (&state);
  queued_view_lifetime (&state);
  multiple_pending_writers (&state);
  concurrent_enqueue_and_view_lifetime (&state);
  same_queue_concurrent_writers (&state);
  request_queue_lifetime (&state);
  pending_export_last_references (&state);
  failed_pending_dependency (&state);
  printf ("actual alias/version/unaligned migration passed: parent=%zu "
          "align=%zu views=%d launches=%d\n",
          state.size, state.alignment, VIEW_COUNT, VIEW_COUNT);
  release_state (&state);
  return 0;
}
