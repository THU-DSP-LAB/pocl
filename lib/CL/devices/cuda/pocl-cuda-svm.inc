void *
pocl_cuda_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemAllocManaged %lu\n", size);
  if ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)
      && ((dev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0))
    {
      POCL_MSG_ERR (
          "This device does not support SVM fine-grained buffers.\n");
      return NULL;
    }

  CUdeviceptr dptr;
  CUresult res;
  res = cuMemAllocManaged (&dptr, size, CU_MEM_ATTACH_GLOBAL);
  CUDA_CHECK_ERROR (res, "pocl_cuda_svm_alloc cuMemAllocManaged");
  if (res != CUDA_SUCCESS)
    dptr = 0;
  return (void *)dptr;
}

void
pocl_cuda_svm_free (cl_device_id dev, void *svm_ptr)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemFree %p\n", svm_ptr);
  CUresult res;
  res = cuMemFree ((CUdeviceptr)svm_ptr);
  CUDA_CHECK_ERROR (res, "pocl_cuda_svm_free cuMemFree");
}

void
pocl_cuda_svm_copy (cl_device_id dev, void *__restrict__ dst,
                    const void *__restrict__ src, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemcpy %p -> %p, %lu bytes\n", src, dst, size);
  CUresult res;
  res = cuMemcpy ((CUdeviceptr)dst, (CUdeviceptr)src, size);
  CUDA_CHECK_ABORT (res, "pocl_cuda_svm_copy cuMemcpy");
}

void
pocl_cuda_svm_copy_async (CUstream stream, void *__restrict__ dst,
                          const void *__restrict__ src, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemcpyAsync %p -> %p, %lu bytes\n", src, dst,
                       size);

  CUresult res;
  res = cuMemcpyAsync ((CUdeviceptr)dst, (CUdeviceptr)src, size, stream);
  CUDA_CHECK_ABORT (res, "cuMemcpyAsync");
}

void
pocl_cuda_svm_fill (cl_device_id dev, void *__restrict__ svm_ptr, size_t size,
                    void *__restrict__ pattern, size_t pattern_size)
{
  POCL_MSG_PRINT_CUDA ("SVM MEMFILL %p \n", svm_ptr);

  pocl_cuda_submit_memfill (0, svm_ptr, size, 0, pattern, pattern_size);
}


cl_int
pocl_cuda_set_kernel_exec_info_ext (cl_device_id dev,
                                     unsigned program_device_i,
                                     cl_kernel Kernel, cl_uint param_name,
                                     size_t param_value_size,
                                     const void *param_value)
{
  pocl_cuda_device_data_t *data = (pocl_cuda_device_data_t *)dev->data;
  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL:
      return data->supports_managed_memory ? CL_SUCCESS : CL_INVALID_OPERATION;
    default:
      POCL_MSG_ERR (
          "CUDA: clSetKernelExecInfo with parameter %u not implemented\n",
          param_name);
      return CL_INVALID_OPERATION;
    }
}

cl_int
pocl_cuda_get_device_info_ext (cl_device_id device, cl_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret)
{
  pocl_cuda_device_data_t *data = (pocl_cuda_device_data_t *)device->data;
  CUdevice cudaDev = data->device;
  int value;
  CUresult res;

  switch (param_name)
    {
    case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_REGISTERS_PER_BLOCK_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_WARP_SIZE_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_GPU_OVERLAP_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_INTEGRATED_MEMORY_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_BUS_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_SLOT_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_DOMAIN_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);

    case CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL:
      return CL_INVALID_VALUE;

    case CL_DEVICE_SUB_GROUP_SIZES_INTEL:
      {
        size_t sizes[] = { data->warp_size };
        POCL_RETURN_GETINFO_ARRAY (size_t, sizeof (sizes) / sizeof (size_t),
                                   sizes);
      }

    default:
      POCL_MSG_ERR ("Unknown param_name for get_device_info_ext: %u\n",
                    param_name);
      return CL_INVALID_VALUE;
    }
}

cl_int
pocl_cuda_get_subgroup_info_ext (cl_device_id dev,
                                 cl_kernel kernel,
                                 unsigned int program_device_i,
                                 cl_kernel_sub_group_info param_name,
                                 size_t input_value_size,
                                 const void *input_value,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret)
{
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)dev->data;
  size_t sg_size = ddata->warp_size;

  switch (param_name)
    {
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
      {
        POCL_RETURN_GETINFO (size_t, sg_size);
      }
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:
      {
        size_t a = (size_t)dev->max_num_sub_groups;
        size_t ndcount[3] = { 1, 1, 1 };
        for (size_t i = 0; i < 3; ++i)
          {
            if (input_value_size / sizeof (size_t) > i)
              ndcount[i] = ((size_t *)input_value)[i];
          }
        size_t b = ndcount[0] * ndcount[1] * ndcount[2];
        b = (b + sg_size - 1) / sg_size;
        POCL_RETURN_GETINFO (size_t, a < b ? a : b);
      }
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:
      {
        POCL_RETURN_ERROR_ON ((input_value == NULL), CL_INVALID_VALUE,
                              "SG size wish not given.");
        size_t n_wish = *(size_t *)input_value;

        size_t nd[3];
        if (n_wish * sg_size > dev->max_work_group_size)
          {
            nd[0] = nd[1] = nd[2] = 0;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof (size_t), nd);
          }
        else
          {
            nd[0] = n_wish * sg_size;
            nd[1] = 1;
            nd[2] = 1;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof (size_t), nd);
          }
      }
    default:
      POCL_RETURN_ERROR_ON (1, CL_INVALID_VALUE, "Unknown param_name: %u\n",
                            param_name);
    }
}
