#include "pocl_migration_queues.h"
#include "pocl_util.h"
#include <stdlib.h>

/* Prepare before publishing command/version state, so resource failures use
 * the public enqueue error path without leaving an unsubmitted last updater.
 * Ordinary context retains keep queues alive if buffer updater events outlive
 * their requesting user queue. */
static void
release_queue_array (cl_command_queue *queues, unsigned count)
{
  if (queues == NULL)
    return;
  for (unsigned i = 0; i < count; ++i)
    if (queues[i] != NULL)
      POname (clReleaseCommandQueue) (queues[i]);
  free (queues);
}

static void
trace_published_queues (cl_command_queue queue)
{
  for (unsigned i = 0; i < queue->context->num_devices; ++i)
    {
      cl_device_id device = queue->context->devices[i];
      POCL_MSG_PRINT_MEMORY (
          "Published migration queue request %" PRIu64 " device %" PRIu64
          " global_mem_id %u context_index %u migration %" PRIu64 "\n",
          queue->id, device->id, (unsigned)device->global_mem_id, i,
          queue->migration_queues[i]->id);
    }
}

cl_int
pocl_prepare_migration_queues (cl_command_queue queue)
{
  POCL_LOCK_OBJ (queue);
  int initialized = queue->migration_queues != NULL;
  POCL_UNLOCK_OBJ (queue);
  if (initialized)
    return CL_SUCCESS;
  const unsigned count = queue->context->num_devices;
  cl_command_queue *candidate = calloc (count, sizeof (cl_command_queue));
  if (candidate == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  cl_int error = CL_SUCCESS;
  int published = 0;
  for (unsigned i = 0; error == CL_SUCCESS && i < count; ++i)
    {
      cl_device_id device = queue->context->devices[i];
      cl_command_queue_properties properties
          = queue->properties & device->on_host_queue_props
            & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
      candidate[i] = POname (clCreateCommandQueue) (queue->context, device,
                                                    properties, &error);
    }
  if (error == CL_SUCCESS)
    {
      POCL_LOCK_OBJ (queue);
      if (queue->migration_queues == NULL)
        {
          queue->migration_queues = candidate;
          candidate = NULL;
          published = 1;
        }
      POCL_UNLOCK_OBJ (queue);
    }
  if (published)
    trace_published_queues (queue);
  release_queue_array (candidate, count);
  return error;
}

cl_command_queue
pocl_migration_queue (cl_command_queue queue, unsigned index)
{
  POCL_LOCK_OBJ (queue);
  assert (queue->migration_queues != NULL);
  cl_command_queue result = queue->migration_queues[index];
  assert (result != NULL);
  POCL_UNLOCK_OBJ (queue);
  return result;
}

void
pocl_release_migration_queues (cl_command_queue queue)
{
  release_queue_array (queue->migration_queues, queue->context->num_devices);
  queue->migration_queues = NULL;
}
