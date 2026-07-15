/* CUDA command queue lifecycle, worker threads, and process-exit cleanup. */

#include "config.h"

#include "common.h"
#include "pocl-cuda-queue.h"
#include "pocl-cuda.h"
#include "pocl_debug.h"
#include "pocl_runtime_config.h"

#include <stdlib.h>

void pocl_cuda_submit_node(_cl_command_node *, cl_command_queue, int);
void pocl_cuda_finalize_command(cl_device_id, cl_event);

enum {
  CUDA_QUEUE_CLEANUP_EXTERNAL = 0,
  CUDA_QUEUE_CLEANUP_SUBMIT_THREAD,
  CUDA_QUEUE_CLEANUP_FINALIZE_THREAD,
};

static pthread_mutex_t active_queues_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t active_queues_cond = PTHREAD_COND_INITIALIZER;
static pocl_cuda_queue_data_t *active_queues;
static pocl_cuda_queue_data_t *cleanup_queues;
static pthread_t cleanup_reaper_thread;
static int cleanup_reaper_started;
static int cleanup_reaper_shutdown;
static int exit_handler_registered;

static void shutdown_active_queues(void);
static void destroy_claimed_queue(pocl_cuda_queue_data_t *);
static void destroy_queue_data(pocl_cuda_queue_data_t *);

static void abort_on_cuda_error(CUresult result, const char *api) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = "unknown CUDA error";
  const char *description = "no CUDA error description";
  const char *queried = NULL;
  if (cuGetErrorName(result, &queried) == CUDA_SUCCESS && queried)
    name = queried;
  queried = NULL;
  if (cuGetErrorString(result, &queried) == CUDA_SUCCESS && queried)
    description = queried;
  POCL_ABORT("CUDA error %d during %s. %s: %s\n", (int)result, api, name,
             description);
}

static int register_exit_handler(void) {
  int status = 0;
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  if (!exit_handler_registered) {
    status = atexit(shutdown_active_queues);
    if (status == 0)
      exit_handler_registered = 1;
  }
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
  return status;
}

static void track_queue(pocl_cuda_queue_data_t *queue_data) {
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  DL_APPEND(active_queues, queue_data);
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
}

static void *cleanup_reaper(void *unused) {
  (void)unused;
  while (1) {
    PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
    while (cleanup_queues == NULL && !cleanup_reaper_shutdown)
      PTHREAD_CHECK(
          pthread_cond_wait(&active_queues_cond, &active_queues_lock));
    pocl_cuda_queue_data_t *queue_data = cleanup_queues;
    if (queue_data == NULL) {
      PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
      return NULL;
    }
    DL_DELETE(cleanup_queues, queue_data);
    PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
    destroy_queue_data(queue_data);
  }
}

static int start_cleanup_reaper(void) {
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  int status = 0;
  if (!cleanup_reaper_started) {
    status = pthread_create(&cleanup_reaper_thread, NULL, cleanup_reaper, NULL);
    if (status == 0)
      cleanup_reaper_started = 1;
  }
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
  return status;
}

static void stop_cleanup_reaper(void) {
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  if (!cleanup_reaper_started) {
    PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
    return;
  }
  cleanup_reaper_shutdown = 1;
  PTHREAD_CHECK(pthread_cond_broadcast(&active_queues_cond));
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
  PTHREAD_CHECK(pthread_join(cleanup_reaper_thread, NULL));
}

static void shutdown_active_queues(void) {
  while (1) {
    PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
    pocl_cuda_queue_data_t *queue_data = active_queues;
    if (queue_data == NULL) {
      PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
      stop_cleanup_reaper();
      return;
    }
    queue_data->exit_cleanup_claimed = 1;
    PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));

    destroy_claimed_queue(queue_data);
  }
}

static void *submit_thread(void *data);
static void *finalize_thread(void *data);

static cl_int start_queue_threads(pocl_cuda_queue_data_t *queue_data) {
  PTHREAD_CHECK(pthread_mutex_init(&queue_data->lock, NULL));
  PTHREAD_CHECK(pthread_cond_init(&queue_data->pending_cond, NULL));
  PTHREAD_CHECK(pthread_cond_init(&queue_data->running_cond, NULL));

  int error = pthread_create(&queue_data->submit_thread, NULL, submit_thread,
                             queue_data);
  if (error) {
    POCL_MSG_ERR("[CUDA] Error creating submit thread: %d\n", error);
    return CL_OUT_OF_RESOURCES;
  }
  queue_data->submit_thread_started = 1;

  error = pthread_create(&queue_data->finalize_thread, NULL, finalize_thread,
                         queue_data);
  if (!error) {
    queue_data->finalize_thread_started = 1;
    return CL_SUCCESS;
  }

  POCL_MSG_ERR("[CUDA] Error creating finalize thread: %d\n", error);
  PTHREAD_CHECK(pthread_mutex_lock(&queue_data->lock));
  queue_data->queue = NULL;
  PTHREAD_CHECK(pthread_cond_broadcast(&queue_data->pending_cond));
  PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
  return CL_OUT_OF_RESOURCES;
}

static void destroy_queue_data(pocl_cuda_queue_data_t *queue_data) {
  if (queue_data->submit_thread_started) {
    if (pthread_equal(pthread_self(), queue_data->submit_thread))
      PTHREAD_CHECK(pthread_detach(queue_data->submit_thread));
    else
      PTHREAD_CHECK(pthread_join(queue_data->submit_thread, NULL));
  }
  if (queue_data->finalize_thread_started) {
    if (pthread_equal(pthread_self(), queue_data->finalize_thread))
      PTHREAD_CHECK(pthread_detach(queue_data->finalize_thread));
    else
      PTHREAD_CHECK(pthread_join(queue_data->finalize_thread, NULL));
  }

  abort_on_cuda_error(cuCtxSetCurrent(queue_data->context), "cuCtxSetCurrent");
  abort_on_cuda_error(cuStreamDestroy(queue_data->stream), "cuStreamDestroy");
  if (queue_data->use_threads) {
    PTHREAD_CHECK(pthread_cond_destroy(&queue_data->running_cond));
    PTHREAD_CHECK(pthread_cond_destroy(&queue_data->pending_cond));
    PTHREAD_CHECK(pthread_mutex_destroy(&queue_data->lock));
  }
  free(queue_data);
}

cl_int pocl_cuda_create_queue(cl_command_queue queue, CUcontext context) {
  abort_on_cuda_error(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  pocl_cuda_queue_data_t *queue_data = calloc(1, sizeof(*queue_data));
  if (!queue_data)
    return CL_OUT_OF_HOST_MEMORY;
  queue_data->context = context;
  queue_data->queue = queue;
  queue->data = queue_data;

  CUresult result = cuStreamCreate(&queue_data->stream, CU_STREAM_NON_BLOCKING);
  if (result != CUDA_SUCCESS) {
    const char *description = NULL;
    cuGetErrorString(result, &description);
    POCL_MSG_ERR("[CUDA] cuStreamCreate failed: %s\n",
                 description ? description : "unknown CUDA error");
    POCL_MEM_FREE(queue->data);
    return CL_OUT_OF_RESOURCES;
  }

  if (register_exit_handler() != 0) {
    POCL_MSG_ERR("[CUDA] Failed to register process-exit queue cleanup\n");
    queue->data = NULL;
    destroy_queue_data(queue_data);
    return CL_OUT_OF_RESOURCES;
  }
  int reaper_error = start_cleanup_reaper();
  if (reaper_error != 0) {
    POCL_MSG_ERR("[CUDA] Error creating queue cleanup reaper: %d\n",
                 reaper_error);
    queue->data = NULL;
    destroy_queue_data(queue_data);
    return CL_OUT_OF_RESOURCES;
  }

  queue_data->use_threads =
      !pocl_get_bool_option("POCL_CUDA_DISABLE_QUEUE_THREADS", 0);
  if (queue_data->use_threads &&
      start_queue_threads(queue_data) != CL_SUCCESS) {
    queue->data = NULL;
    destroy_queue_data(queue_data);
    return CL_OUT_OF_RESOURCES;
  }

  track_queue(queue_data);
  return CL_SUCCESS;
}

static int current_cleanup_owner(pocl_cuda_queue_data_t *queue_data) {
  int cleanup_owner = CUDA_QUEUE_CLEANUP_EXTERNAL;
  if (queue_data->use_threads &&
      pthread_equal(pthread_self(), queue_data->submit_thread))
    cleanup_owner = CUDA_QUEUE_CLEANUP_SUBMIT_THREAD;
  else if (queue_data->use_threads &&
           pthread_equal(pthread_self(), queue_data->finalize_thread))
    cleanup_owner = CUDA_QUEUE_CLEANUP_FINALIZE_THREAD;
  return cleanup_owner;
}

static void stop_queue_threads(pocl_cuda_queue_data_t *queue_data) {
  if (queue_data->use_threads) {
    PTHREAD_CHECK(pthread_mutex_lock(&queue_data->lock));
    assert(queue_data->pending_queue == NULL);
    assert(queue_data->running_queue == NULL);
    queue_data->queue = NULL;
    PTHREAD_CHECK(pthread_cond_broadcast(&queue_data->pending_cond));
    PTHREAD_CHECK(pthread_cond_broadcast(&queue_data->running_cond));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
  } else {
    assert(queue_data->pending_queue == NULL);
    assert(queue_data->running_queue == NULL);
  }
}

static pocl_cuda_queue_data_t *take_queue_for_release(
    cl_command_queue queue, int *cleanup_owner) {
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  pocl_cuda_queue_data_t *queue_data = queue->data;
  if (queue_data == NULL) {
    PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
    return NULL;
  }
  if (queue_data->exit_cleanup_claimed) {
    while (queue->data == queue_data)
      PTHREAD_CHECK(
          pthread_cond_wait(&active_queues_cond, &active_queues_lock));
    PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
    return NULL;
  }
  *cleanup_owner = current_cleanup_owner(queue_data);
  DL_DELETE(active_queues, queue_data);
  if (*cleanup_owner != CUDA_QUEUE_CLEANUP_EXTERNAL) {
    queue_data->prev = NULL;
    queue_data->next = NULL;
    DL_APPEND(cleanup_queues, queue_data);
    PTHREAD_CHECK(pthread_cond_signal(&active_queues_cond));
  }
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
  return queue_data;
}

static void destroy_claimed_queue(pocl_cuda_queue_data_t *queue_data) {
  cl_command_queue queue = queue_data->queue;
  POCL_LOCK_OBJ(queue);
  cl_event event = queue->last_event.event;
  if (event)
    POname(clRetainEvent)(event);
  POCL_UNLOCK_OBJ(queue);
  if (event)
    pocl_cuda_wait_event(queue->device, event);

  stop_queue_threads(queue_data);
  PTHREAD_CHECK(pthread_mutex_lock(&active_queues_lock));
  DL_DELETE(active_queues, queue_data);
  queue->data = NULL;
  PTHREAD_CHECK(pthread_cond_broadcast(&active_queues_cond));
  PTHREAD_CHECK(pthread_mutex_unlock(&active_queues_lock));
  destroy_queue_data(queue_data);
  if (event)
    POname(clReleaseEvent)(event);
}

int pocl_cuda_free_queue(cl_device_id device, cl_command_queue queue) {
  (void)device;
  int cleanup_owner = CUDA_QUEUE_CLEANUP_EXTERNAL;
  pocl_cuda_queue_data_t *queue_data =
      take_queue_for_release(queue, &cleanup_owner);
  if (queue_data == NULL)
    return CL_SUCCESS;

  stop_queue_threads(queue_data);
  queue->data = NULL;
  if (cleanup_owner == CUDA_QUEUE_CLEANUP_EXTERNAL)
    destroy_queue_data(queue_data);
  return CL_SUCCESS;
}

void pocl_cuda_join(cl_device_id device, cl_command_queue queue) {
  POCL_LOCK_OBJ(queue);
  cl_event event = queue->last_event.event;
  if (!event) {
    POCL_UNLOCK_OBJ(queue);
    return;
  }
  POname(clRetainEvent)(event);
  POCL_UNLOCK_OBJ(queue);
  pocl_cuda_wait_event(device, event);
  POname(clReleaseEvent)(event);
}

static void *submit_thread(void *data) {
  pocl_cuda_queue_data_t *queue_data = data;
  if (!queue_data->queue)
    return NULL;
  abort_on_cuda_error(cuCtxSetCurrent(queue_data->context), "cuCtxSetCurrent");

  while (1) {
    _cl_command_node *node = NULL;
    PTHREAD_CHECK(pthread_mutex_lock(&queue_data->lock));
    if (!queue_data->queue) {
      PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
      break;
    }
    if (!queue_data->pending_queue)
      PTHREAD_CHECK(
          pthread_cond_wait(&queue_data->pending_cond, &queue_data->lock));
    if (queue_data->pending_queue) {
      node = queue_data->pending_queue;
      DL_DELETE(queue_data->pending_queue, node);
    }
    PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
    if (!node)
      continue;

    pocl_cuda_submit_node(node, queue_data->queue, 0);
    PTHREAD_CHECK(pthread_mutex_lock(&queue_data->lock));
    DL_APPEND(queue_data->running_queue, node);
    PTHREAD_CHECK(pthread_cond_signal(&queue_data->running_cond));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
  }

  return NULL;
}

static void *finalize_thread(void *data) {
  pocl_cuda_queue_data_t *queue_data = data;
  cl_command_queue queue = queue_data->queue;
  if (!queue)
    return NULL;
  abort_on_cuda_error(cuCtxSetCurrent(queue_data->context), "cuCtxSetCurrent");

  while (1) {
    _cl_command_node *node = NULL;
    PTHREAD_CHECK(pthread_mutex_lock(&queue_data->lock));
    if (!queue_data->queue) {
      PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
      break;
    }
    if (!queue_data->running_queue)
      PTHREAD_CHECK(
          pthread_cond_wait(&queue_data->running_cond, &queue_data->lock));
    if (queue_data->running_queue) {
      node = queue_data->running_queue;
      DL_DELETE(queue_data->running_queue, node);
    }
    PTHREAD_CHECK(pthread_mutex_unlock(&queue_data->lock));
    if (node)
      pocl_cuda_finalize_command(queue->device, node->sync.event.event);
  }

  return NULL;
}
