static pocl_cuda_host_registration_t
pocl_cuda_register_host_memory (cl_mem mem)
{
  pocl_cuda_host_registration_t registration = {
    .status = CL_MEM_OBJECT_ALLOCATION_FAILURE,
  };
  if (pocl_alloc_or_retain_mem_host_ptr (mem) != 0)
    {
      registration.status = CL_OUT_OF_HOST_MEMORY;
      return registration;
    }

  CUresult result = cuMemHostRegister (mem->mem_host_ptr, mem->size,
                                       CU_MEMHOSTREGISTER_DEVICEMAP);
  bool owns_registration = result == CUDA_SUCCESS;
  if (!owns_registration
      && result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
    {
      CUDA_CHECK_ERROR (result, "cuMemHostRegister");
      pocl_release_mem_host_ptr (mem);
      return registration;
    }

  CUdeviceptr device_base = 0;
  result = cuMemHostGetDevicePointer (&device_base, mem->mem_host_ptr, 0);
  if (result != CUDA_SUCCESS)
    {
      CUDA_CHECK_ERROR (result, "cuMemHostGetDevicePointer");
      if (owns_registration)
        cuMemHostUnregister (mem->mem_host_ptr);
      pocl_release_mem_host_ptr (mem);
      return registration;
    }

  registration.status = CL_SUCCESS;
  registration.device_pointer = (void *)device_base;
  registration.registration_base = mem->mem_host_ptr;
  registration.owns_registration = owns_registration
                                       ? CUDA_HOST_REGISTRATION_OWNED
                                       : CUDA_HOST_REGISTRATION_NOT_OWNED;
  return registration;
}

cl_int
pocl_cuda_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  CUresult result;
  void *b = NULL;

  p->extra_ptr = NULL;
  p->extra = CUDA_HOST_REGISTRATION_NOT_OWNED;
  p->mem_ptr = NULL;
  p->version = 0;
  cl_mem_flags flags = mem->flags;

  if (flags & CL_MEM_USE_HOST_PTR)
    {
      if (mem->mem_host_ptr_is_svm)
        {
          assert (device->svm_caps & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                                      | CL_DEVICE_SVM_FINE_GRAIN_BUFFER));

          p->version = mem->mem_host_ptr_version;
          p->mem_ptr = mem->mem_host_ptr;
          p->device_addr = p->mem_ptr;
          return CL_SUCCESS;
        }

      if (!((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
        {
          /* cuMemHostRegister is not supported on some ARM devices like the Nano, but supported on Xavier.
           * Allocate device memory and perform explicit copies
           * before and after running a kernel */
          result = cuMemAlloc ((CUdeviceptr *)&b, mem->size);
          if (CUDA_CHECK_ERROR (result, "cuMemAlloc"))
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }
      else
        {
          pocl_cuda_host_registration_t registration
              = pocl_cuda_register_host_memory (mem);
          if (registration.status != CL_SUCCESS)
            return registration.status;
          b = registration.device_pointer;
          p->extra_ptr = registration.registration_base;
          p->extra = registration.owns_registration;

          /* TODO can we assume cuMemHostRegister copies
           * the content of host memory to the device ? for now, lets not */
          p->version = 0;
        }
    }
  /* preallocate host visible memory */
  else if ((flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    {
      result = cuMemHostAlloc (&p->extra_ptr, mem->size,
                               CU_MEMHOSTREGISTER_DEVICEMAP);
      if (CUDA_CHECK_ERROR (result, "cuMemHostAlloc"))
        {
          p->extra_ptr = NULL;
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b, p->extra_ptr, 0);
      if (CUDA_CHECK_ERROR (result, "cuMemHostGetDevicePointer"))
        {
          cuMemFreeHost (p->extra_ptr);
          p->extra_ptr = NULL;
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      mem->mem_host_ptr = p->extra_ptr;
      mem->mem_host_ptr_refcount = 1;
      mem->mem_host_ptr_version = 0;

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          result = cuMemcpyHtoD ((CUdeviceptr)b, host_ptr, mem->size);
          if (CUDA_CHECK_ERROR (result, "cuMemcpyHtoD"))
            {
              cuMemFreeHost (p->extra_ptr);
              p->extra_ptr = NULL;
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }

          result = cuStreamSynchronize (0);
          if (CUDA_CHECK_ERROR (result, "cuStreamSynchronize"))
            {
              cuMemFreeHost (p->extra_ptr);
              p->extra_ptr = NULL;
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }

          mem->mem_host_ptr_version = 1;
          mem->latest_version = 1;
          p->version = 1;
        }
    }
  else
    {
      result = cuMemAlloc ((CUdeviceptr *)&b, mem->size);
      if (CUDA_CHECK_ERROR (result, "cuMemAlloc"))
        {
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }
    }

  p->mem_ptr = b;
  return CL_SUCCESS;
}

void
pocl_cuda_free (cl_device_id device, cl_mem mem_obj)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];

  if (p->extra_ptr != NULL)
    {
      CUresult result = cuCtxSynchronize ();
      CUDA_CHECK_ABORT (result, "cuCtxSynchronize before freeing host memory");
    }

  if (mem_obj->flags & CL_MEM_USE_HOST_PTR)
    {
      if (mem_obj->mem_host_ptr_is_svm)
        {
          p->device_addr = NULL;
        }
      else if (((pocl_cuda_device_data_t *)device->data)
                   ->supports_cu_mem_host_register)
        {
          if (p->extra == CUDA_HOST_REGISTRATION_OWNED)
            {
              CUresult result = cuMemHostUnregister (p->extra_ptr);
              CUDA_CHECK_ERROR (result, "cuMemHostUnregister");
            }
          pocl_release_mem_host_ptr (mem_obj);
          p->extra = CUDA_HOST_REGISTRATION_NOT_OWNED;
          p->extra_ptr = NULL;
        }
      else
        {
          cuMemFree ((CUdeviceptr)p->mem_ptr);
        }
    }
  else if (p->extra_ptr)
    {
      mem_obj->mem_host_ptr = NULL;
      mem_obj->mem_host_ptr_refcount = 0;
      mem_obj->mem_host_ptr_version = 0;
      cuMemFreeHost (p->extra_ptr);
      p->extra_ptr = NULL;
    }
  else
    {
      assert (p->extra_ptr == NULL);
      assert (p->mem_ptr != NULL);
      cuMemFree ((CUdeviceptr)p->mem_ptr);
    }
  p->mem_ptr = NULL;
  p->version = 0;
}

int
pocl_cuda_can_migrate_d2d (cl_device_id dest, cl_device_id source)
{
  assert (dest != source);
  if (strcmp (dest->ops->device_name, source->ops->device_name) == 0)
    {
      int possible;
      pocl_cuda_device_data_t *src_dev
          = (pocl_cuda_device_data_t *)source->data;
      pocl_cuda_device_data_t *dst_dev = (pocl_cuda_device_data_t *)dest->data;
      cuDeviceCanAccessPeer (&possible, src_dev->device, dst_dev->device);
      POCL_MSG_PRINT_CUDA ("cuDeviceCanAccessPeer %p and %p -> %s\n", source,
                           dest, possible ? "yes" : "no");
      if (possible)
        {
          cuCtxSetCurrent (dst_dev->context);
          CUresult res = cuCtxEnablePeerAccess (src_dev->context, 0);
          POCL_MSG_PRINT_CUDA ("cuCtxEnablePeerAccess from %p to %p : %u\n",
                               source, dest, res);
          if (res == CUDA_SUCCESS
              || res == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            return 1;
        }
    }

  return 0;
}
