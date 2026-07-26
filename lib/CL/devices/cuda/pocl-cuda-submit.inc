void
pocl_cuda_submit_node (_cl_command_node *node, cl_command_queue cq, int locked)
{
  CUresult result;
  CUstream stream = ((pocl_cuda_queue_data_t *)cq->data)->stream;

  if (!locked)
    POCL_LOCK_OBJ (node->sync.event.event);

  pocl_cuda_event_data_t *event_data
      = (pocl_cuda_event_data_t *)node->sync.event.event->data;

  /* Process event dependencies */
  event_node *dep = NULL;
  LL_FOREACH (node->sync.event.event->wait_list, dep)
  {
    /* If it is in the process of completing, just skip it */
    if (dep->event->status <= CL_COMPLETE)
      continue;

    /* Add CUDA event dependency */
    if (dep->event->command_type != CL_COMMAND_USER
        && dep->event->queue->device->ops == cq->device->ops)
      {
        /* Block stream on event, but only for different queues */
        if (dep->event->queue != node->sync.event.event->queue)
          {
            pocl_cuda_event_data_t *dep_data
                = (pocl_cuda_event_data_t *)dep->event->data;

            /* Wait until dependency has finished being submitted */
            while (!dep_data->events_ready)
              ;

            result = cuStreamWaitEvent (stream, dep_data->end, 0);
            CUDA_CHECK_ABORT (result, "cuStreamWaitEvent");
          }
      }
    else
      {
        if (!((pocl_cuda_queue_data_t *)cq->data)->use_threads)
          POCL_ABORT (
              "Can't handle non-CUDA dependencies without queue threads\n");

        event_data->num_ext_events++;
      }
    }

  /* CUDA async copies involving pageable host memory may block in the driver
   * when they are enqueued behind an unresolved stream wait. Waiting for
   * external dependencies on the host keeps the submit thread out of CUDA
   * until the user or foreign-device event is complete. The event lock and
   * condition variable make the dependency count immune to missed wakeups. */
  if (event_data->num_ext_events)
    {
      if (!((pocl_cuda_queue_data_t *)cq->data)->use_threads)
        POCL_ABORT (
            "Can't handle non-CUDA dependencies without queue threads\n");
      while (event_data->num_ext_events)
        PTHREAD_CHECK (pthread_cond_wait (&event_data->event_cond,
                                          &node->sync.event.event->pocl_lock));
    }

  /* Create and record event for command start if profiling enabled */
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    {
      result = cuEventCreate (&event_data->start, CU_EVENT_DEFAULT);
      CUDA_CHECK_ABORT (result, "cuEventCreate");
      result = cuEventRecord (event_data->start, stream);
      CUDA_CHECK_ABORT (result, "cuEventRecord");
    }

  pocl_update_event_submitted (node->sync.event.event);

  POCL_UNLOCK_OBJ (node->sync.event.event);

  cl_event event = node->sync.event.event;
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;

  /* If a wait-list dependency has already failed, this command must not run:
   * skip issuing any device work for it (its inputs may have been aborted), but
   * still record the end event below so pocl_cuda_finalize_command can sync and
   * clean up on valid CUDA handles. finalize_command honors failed_dependency
   * and terminates the command via the error cascade. Unlike the other drivers
   * CUDA does not call pocl_update_event_failed from submit: its finalize path
   * already drives the failure, and bailing out here would leave the end event
   * unrecorded (finalize would cuEventSynchronize on a garbage handle). */
  if (!event->failed_dependency)
  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      pocl_cuda_submit_read (
        stream, cmd->read.dst_host_ptr,
        &cmd->read.src->device_ptrs[dev->global_mem_id],
        node->command.read.offset, node->command.read.size);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      pocl_cuda_submit_write (
        stream, cmd->write.src_host_ptr,
        &cmd->write.dst->device_ptrs[dev->global_mem_id],
        node->command.write.offset, node->command.write.size);
      break;
    case CL_COMMAND_COPY_BUFFER:
      {
        pocl_cuda_submit_copy (
          stream, &cmd->copy.src->device_ptrs[dev->global_mem_id],
          cmd->copy.src_offset,
          &cmd->copy.dst->device_ptrs[dev->global_mem_id],
          cmd->copy.dst_offset, cmd->copy.size);
        break;
      }
    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_cuda_submit_read_rect (
        stream, cmd->read_rect.dst_host_ptr,
        cmd->read_rect.src->device_ptrs[dev->global_mem_id].mem_ptr,
        cmd->read_rect.buffer_origin, cmd->read_rect.host_origin,
        cmd->read_rect.region, cmd->read_rect.buffer_row_pitch,
        cmd->read_rect.buffer_slice_pitch, cmd->read_rect.host_row_pitch,
        cmd->read_rect.host_slice_pitch);
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_cuda_submit_write_rect (
        stream, cmd->write_rect.src_host_ptr,
        cmd->write_rect.dst->device_ptrs[dev->global_mem_id].mem_ptr,
        cmd->write_rect.buffer_origin, cmd->write_rect.host_origin,
        cmd->write_rect.region, cmd->write_rect.buffer_row_pitch,
        cmd->write_rect.buffer_slice_pitch, cmd->write_rect.host_row_pitch,
        cmd->write_rect.host_slice_pitch);
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      {
        pocl_cuda_submit_copy_rect (
          stream, dev,
          cmd->copy_rect.src->device_ptrs[dev->global_mem_id].mem_ptr,
          cmd->copy_rect.dst->device_ptrs[dev->global_mem_id].mem_ptr,
          cmd->copy_rect.src_origin, cmd->copy_rect.dst_origin,
          cmd->copy_rect.region, cmd->copy_rect.src_row_pitch,
          cmd->copy_rect.src_slice_pitch, cmd->copy_rect.dst_row_pitch,
          cmd->copy_rect.dst_slice_pitch);
        break;
      }
    case CL_COMMAND_MAP_BUFFER:
      {
        pocl_cuda_submit_map_mem (
          stream, &cmd->map.buffer->device_ptrs[dev->global_mem_id],
          cmd->map.mapping->offset, cmd->map.mapping->size,
          cmd->map.mapping->host_ptr);
        break;
      }
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      {
        cl_mem buffer = cmd->unmap.buffer;
        assert (buffer->is_image == CL_FALSE);
        pocl_cuda_submit_unmap_mem (
          stream, &cmd->unmap.buffer->device_ptrs[dev->global_mem_id],
          cmd->unmap.mapping->offset, cmd->unmap.mapping->size,
          cmd->unmap.mapping->host_ptr, cmd->unmap.mapping->map_flags);
        break;
      }
    case CL_COMMAND_NDRANGE_KERNEL:
      pocl_cuda_submit_kernel (stream, node, node->device,
                               node->sync.event.event);
      break;

    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      switch (cmd->migrate.type)
        {
        case ENQUEUE_MIGRATE_TYPE_D2H:
          {
            cl_mem mem = node->migr_infos->buffer;
            pocl_cuda_submit_read (
              stream, mem->mem_host_ptr,
              &mem->device_ptrs[dev->global_mem_id], 0, mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_H2D:
          {
            cl_mem mem = node->migr_infos->buffer;
            pocl_cuda_submit_write (
              stream, mem->mem_host_ptr,
              &mem->device_ptrs[dev->global_mem_id], 0, mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_D2D:
          {
            cl_mem mem = node->migr_infos->buffer;
            pocl_cuda_submit_copy_p2p (
              stream, cmd->migrate.src_device,
              mem->device_ptrs[cmd->migrate.src_device->global_mem_id].mem_ptr,
              0, cq->device, mem->device_ptrs[dev->global_mem_id].mem_ptr, 0,
              mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_NOP:
          {
            break;
          }
        }
      break;

    case CL_COMMAND_MARKER:
    case CL_COMMAND_BARRIER:
      break;

    case CL_COMMAND_FILL_BUFFER:
      pocl_cuda_submit_memfill (
        stream, cmd->memfill.dst->device_ptrs[dev->global_mem_id].mem_ptr,
        cmd->memfill.size, cmd->memfill.offset, cmd->memfill.pattern,
        cmd->memfill.pattern_size);
      break;
    case CL_COMMAND_SVM_MAP:
    case CL_COMMAND_SVM_UNMAP:
      /* empty */
      break;
    case CL_COMMAND_SVM_MIGRATE_MEM:
      /* CUDA managed allocations migrate on demand. Recording this command's
         stream event preserves the OpenCL queue and event dependencies. */
      break;

    case CL_COMMAND_SVM_MEMCPY:
      pocl_cuda_svm_copy_async (stream, cmd->svm_memcpy.dst,
                                cmd->svm_memcpy.src, cmd->svm_memcpy.size);
      break;
    case CL_COMMAND_SVM_MEMFILL:
      pocl_cuda_submit_memfill (stream, cmd->svm_fill.svm_ptr,
                                cmd->svm_fill.size, 0, cmd->svm_fill.pattern,
                                cmd->svm_fill.pattern_size);
      break;
    case CL_COMMAND_SVM_FREE:
      if (cmd->svm_free.pfn_free_func)
        {
          cmd->svm_free.pfn_free_func (
              cmd->svm_free.queue, cmd->svm_free.num_svm_pointers,
              cmd->svm_free.svm_pointers, cmd->svm_free.data);
        }
      else
        {
          for (unsigned int i = 0; i < cmd->svm_free.num_svm_pointers; i++)
            {
	      void *ptr = cmd->svm_free.svm_pointers[i];
	      /* This updates bookkeeping associated with the 'ptr'
		 done by the PoCL core. */
	      POname (clSVMFree) (event->context, ptr);
	    }
        }
      break;
    case CL_COMMAND_READ_IMAGE:
    case CL_COMMAND_WRITE_IMAGE:
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_FILL_IMAGE:
    case CL_COMMAND_MAP_IMAGE:
    case CL_COMMAND_NATIVE_KERNEL:
    default:
      POCL_ABORT_UNIMPLEMENTED (pocl_command_to_str (node->type));
      break;
    }

  /* Create and record event for command end */
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    result = cuEventCreate (&event_data->end, CU_EVENT_DEFAULT);
  else
    result = cuEventCreate (&event_data->end, CU_EVENT_DISABLE_TIMING);
  CUDA_CHECK_ABORT (result, "cuEventCreate");
  result = cuEventRecord (event_data->end, stream);
  CUDA_CHECK_ABORT (result, "cuEventRecord");

  event_data->events_ready = 1;
}
