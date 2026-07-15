/* Internal CUDA command queue state and lifecycle entry points. */

#ifndef POCL_CUDA_QUEUE_H
#define POCL_CUDA_QUEUE_H

#include "pocl_cl.h"

#include <cuda.h>
#include <pthread.h>

typedef struct pocl_cuda_queue_data_s {
  CUcontext context;
  CUstream stream;
  int use_threads;
  int submit_thread_started;
  int finalize_thread_started;
  pthread_t submit_thread;
  pthread_t finalize_thread;
  pthread_mutex_t lock;
  pthread_cond_t pending_cond;
  pthread_cond_t running_cond;
  _cl_command_node *volatile pending_queue;
  _cl_command_node *volatile running_queue;
  cl_command_queue queue;
  int exit_cleanup_claimed;
  struct pocl_cuda_queue_data_s *prev;
  struct pocl_cuda_queue_data_s *next;
} pocl_cuda_queue_data_t;

cl_int pocl_cuda_create_queue(cl_command_queue queue, CUcontext context);

#endif
