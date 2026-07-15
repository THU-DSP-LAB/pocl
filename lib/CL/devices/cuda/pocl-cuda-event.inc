void
pocl_cuda_submit (_cl_command_node *node, cl_command_queue cq)
{
  /* Allocate CUDA event data */
  pocl_cuda_event_data_t *p
      = (pocl_cuda_event_data_t *)calloc (1, sizeof (pocl_cuda_event_data_t));
  node->sync.event.event->data = p;
  pocl_cuda_queue_data_t *queue_data = (pocl_cuda_queue_data_t *)cq->data;
  p->use_threads = queue_data->use_threads;

  if (p->use_threads)
    {

      PTHREAD_CHECK (pthread_cond_init (&p->event_cond, NULL));
      /* Add command to work queue */
      POCL_UNLOCK_OBJ (node->sync.event.event);
      PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
      DL_APPEND (queue_data->pending_queue, node);
      PTHREAD_CHECK (pthread_cond_signal (&queue_data->pending_cond));
      PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
    }
  else
    {
      /* Submit command in this thread */
      cuCtxSetCurrent (((pocl_cuda_device_data_t *)cq->device->data)->context);
      pocl_cuda_submit_node (node, cq, 1);
    }
}

void
pocl_cuda_notify (cl_device_id device, cl_event event, cl_event finished)
{
  /* Ignore CUDA device events, we've already handled these dependencies */
  if (finished->queue && finished->queue->device->ops == device->ops)
    return;

  if (event->status == CL_QUEUED)
    return;

  pocl_cuda_event_data_t *event_data = (pocl_cuda_event_data_t *)event->data;

  assert (event_data);
  assert (event_data->num_ext_events > 0);
  assert (event_data->ext_event_flag);

  /* If dependency failed, so should we */
  /* TODO: This isn't true if this is an implicit dependency */
  if (finished->status < 0)
    event->status = -1;

  /* Decrement external event counter */
  /* Trigger flag if none left */
  if (!--event_data->num_ext_events)
    *event_data->ext_event_flag = 1;
}

void
pocl_cuda_flush (cl_device_id device, cl_command_queue cq)
{
  /* TODO: Something here? */
}

void
pocl_cuda_finalize_command (cl_device_id device, cl_event event)
{
  CUresult result;
  pocl_cuda_event_data_t *event_data = (pocl_cuda_event_data_t *)event->data;

  /* Wait for command to finish */
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  result = cuEventSynchronize (event_data->end);
  CUDA_CHECK_ABORT (result, "cuEventSynchronize");

  if (event->command_type == CL_COMMAND_NDRANGE_KERNEL
      || event->command_type == CL_COMMAND_TASK)
    {
      if (!((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
        {
          /* On ARM with USE_HOST_PTR, perform explict copies back from device */
          cl_kernel kernel = event->command->command.run.kernel;
          pocl_argument *arguments = event->command->command.run.arguments;
          unsigned i;
          pocl_kernel_metadata_t *meta = kernel->meta;
          for (i = 0; i < meta->num_args; i++)
            {
              if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
                {
                  if (!ARG_IS_LOCAL (meta->arg_info[i]) && arguments[i].value)
                    {
                      cl_mem mem = *(void **)arguments[i].value;
                      if (mem->flags & CL_MEM_USE_HOST_PTR)
                        {
                          CUdeviceptr ptr
                              = (CUdeviceptr)mem->device_ptrs[device->global_mem_id]
                                    .mem_ptr;
                          cuMemcpyDtoH (mem->mem_host_ptr, ptr, mem->size);
                          cuStreamSynchronize (0);
                        }
                    }
                }
            }
        }
    }

  /* Handle failed events */

  /* Read the failed-dependency flag before pocl_update_event_running, which
   * transitions the event to CL_RUNNING and so would mask a status that
   * pocl_cuda_notify set to a negative value. A failed dependency means the
   * command was not issued (see pocl_cuda_submit_node) and must be terminated. */
  int failed_dependency = event->failed_dependency;

  pocl_update_event_running (event);
  if (event->status < 0 || failed_dependency)
    POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, event, "CUDA event failed");
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG (event, "CUDA event");
}

void
pocl_cuda_update_event (cl_device_id device, cl_event event)
{
  if ((event->status == CL_COMPLETE)
      && (event->queue->properties & CL_QUEUE_PROFILING_ENABLE))
    {
      /* Update timing info with CUDA event timers if profiling enabled */
      /* CUDA doesn't provide a way to get event timestamps directly,
       * only the elapsed time between two events. We use the elapsed
       * time from the epoch event enqueued on device creation to get
       * the actual timestamps.
       * More specifically, we measure the time between the epoch event
       * and the start event. This results in the start time. Then we
       * take the elapsed time between the start and end event to
       * compute the end time. Measuring the end time relative to the
       * epoch event may result in unprecise measurements due to the
       * usage of float by CUDA. */

      float diff;
      CUresult result, result2;
      pocl_cuda_event_data_t *event_data
          = (pocl_cuda_event_data_t *)event->data;
      cl_ulong epoch = ((pocl_cuda_device_data_t *)device->data)->epoch;

      result = cuEventElapsedTime (
          &diff, ((pocl_cuda_device_data_t *)device->data)->epoch_event,
          event_data->start);
      CUDA_CHECK_ERROR (result, "cuEventElapsedTime");
      event->time_start
          = epoch + (cl_ulong)(diff * CUDA_NANOSECONDS_PER_MILLISECOND);
      if (event->time_start < event->time_submit)
        event->time_start = event->time_submit;

      result2 = cuEventElapsedTime (&diff, event_data->start, event_data->end);
      CUDA_CHECK_ERROR (result2, "cuEventElapsedTime");
      event->time_end = event->time_start
                        + (cl_ulong)(diff * CUDA_NANOSECONDS_PER_MILLISECOND);

      if (result != CUDA_SUCCESS || result2 != CUDA_SUCCESS)
        {
          event->time_start = 0;
          event->time_end = 0;
        }
    }
}

void
pocl_cuda_wait_event_recurse (cl_device_id device, cl_event event)
{
  while (event->wait_list)
    pocl_cuda_wait_event_recurse (device, event->wait_list->event);

  if (event->status > CL_COMPLETE)
    pocl_cuda_finalize_command (device, event);
}

void
pocl_cuda_notify_event_finished (cl_event event)
{
  pocl_cuda_event_data_t *e_d = (pocl_cuda_event_data_t *)event->data;

  if (e_d->use_threads)
    PTHREAD_CHECK (pthread_cond_broadcast (&e_d->event_cond));
}

void
pocl_cuda_wait_event (cl_device_id device, cl_event event)
{
  pocl_cuda_event_data_t *e_d = (pocl_cuda_event_data_t *)event->data;

  if (e_d->use_threads)
    {
      /* Wait until background thread marks command as complete */
      POCL_LOCK_OBJ (event);
      while (event->status > CL_COMPLETE)
        {
          PTHREAD_CHECK (
              pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
        }
      POCL_UNLOCK_OBJ (event);
    }
  else
    {
      /* Recursively finalize commands in this thread */
      pocl_cuda_wait_event_recurse (device, event);
    }
}

void
pocl_cuda_free_event_data (cl_event event)
{
  if (event->data)
    {
      pocl_cuda_event_data_t *event_data
          = (pocl_cuda_event_data_t *)event->data;
      if (event_data->use_threads)
        PTHREAD_CHECK (pthread_cond_destroy (&event_data->event_cond));
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        cuEventDestroy (event_data->start);
      cuEventDestroy (event_data->end);
      if (event_data->ext_event_flag)
        {
          CUresult result = cuMemFreeHost (event_data->ext_event_flag);
          CUDA_CHECK_ERROR (result, "pocl_cuda_free_event_data cuMemFreeHost");
        }
      free (event->data);
    }
}

char* pocl_cuda_init_build(void *data)
{
    return strdup("-mllvm --nvptx-short-ptr");
}

/****** SVM callbacks *****/
