enum
{
  POCL_CUDA_KERNEL_PARAMETER_BYTES = 4352,
  POCL_CUDA_IMPLICIT_PARAMETER_COUNT = 16,
};

void
pocl_cuda_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "cuda";
  ops->build_hash = pocl_cuda_build_hash;
  ops->probe = pocl_cuda_probe;
  ops->uninit = pocl_cuda_uninit;
  ops->reinit = NULL;
  ops->init = pocl_cuda_init;
  ops->init_queue = pocl_cuda_init_queue;
  ops->free_queue = pocl_cuda_free_queue;

  ops->alloc_mem_obj = pocl_cuda_alloc_mem_obj;
  ops->alloc_mem_obj_at_create = 1;
  ops->free = pocl_cuda_free;

  ops->submit = pocl_cuda_submit;
  ops->notify = pocl_cuda_notify;
  ops->broadcast = pocl_broadcast;
  ops->wait_event = pocl_cuda_wait_event;
  ops->update_event = pocl_cuda_update_event;
  ops->get_synchronized_timestamps = pocl_driver_get_synchronized_timestamps;
  ops->free_event_data = pocl_cuda_free_event_data;
  ops->join = pocl_cuda_join;
  ops->flush = pocl_cuda_flush;
  ops->init_build = pocl_cuda_init_build;
  // TODO
  ops->notify_event_finished = pocl_cuda_notify_event_finished;

  ops->get_device_info_ext = pocl_cuda_get_device_info_ext;
  ops->set_kernel_exec_info_ext = pocl_cuda_set_kernel_exec_info_ext;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;

  ops->post_build_program = pocl_cuda_post_build_program;
  ops->free_program = pocl_cuda_free_program;
  ops->build_builtin = pocl_cuda_build_builtin;

  ops->compile_kernel = pocl_cuda_compile_kernel;
  ops->create_kernel = pocl_cuda_create_kernel;
  ops->free_kernel = pocl_cuda_free_kernel;

  // TODO
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->can_migrate_d2d = pocl_cuda_can_migrate_d2d;
  ops->migrate_d2d = NULL;
  ops->read = NULL;
  ops->read_rect = NULL;
  ops->write = NULL;
  ops->write_rect = NULL;
  ops->copy = NULL;
  ops->copy_rect = NULL;
  ops->map_mem = NULL;
  ops->unmap_mem = NULL;
  ops->run = NULL;

  ops->svm_alloc = pocl_cuda_svm_alloc;
  ops->svm_free = pocl_cuda_svm_free;
  /* No need to implement these two as they are no-ops
   * and pocl_exec_command takes care of them. */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_cuda_svm_copy;
  ops->svm_fill = pocl_cuda_svm_fill;

  ops->get_subgroup_info_ext = pocl_cuda_get_subgroup_info_ext;
}

cl_int
pocl_cuda_init (unsigned j, cl_device_id dev, const char *parameters)
{
  CUresult result;
  int ret = CL_SUCCESS;

  assert (dev->data == NULL);

  dev->llvm_target_triplet = (sizeof (void *) == 8) ? "nvptx64" : "nvptx";
  dev->llvm_cpu = "generic";
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "cuda";

  pocl_init_default_device_infos (dev, CUDA_DEVICE_EXTENSIONS);

  SETUP_DEVICE_CL_VERSION (dev, CUDA_DEVICE_CL_VERSION_MAJOR,
                           CUDA_DEVICE_CL_VERSION_MINOR);

  dev->vendor = "NVIDIA Corporation";
  dev->vendor_id = 0x10de; /* the PCIID for NVIDIA */

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->address_bits = (sizeof (void *) * 8);

  dev->llvm_fp_contract_mode = "fast";

  dev->spmd = CL_TRUE;
  dev->run_workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;

  dev->global_as_id = 1;
  dev->local_as_id = 3;
  dev->constant_as_id = 4;
  dev->device_aux_functions = cuda_native_device_aux_funcs;

  /* TODO: Get images working */
  dev->image_support = CL_FALSE;
  dev->max_read_image_args = 0;
  dev->max_write_image_args = 0;
  dev->max_read_write_image_args = 0;
  dev->num_image_formats[0] = 0;
  dev->num_image_formats[1] = 0;
  dev->num_image_formats[2] = 0;
  dev->num_image_formats[3] = 0;
  dev->num_image_formats[4] = 0;
  dev->num_image_formats[5] = 0;
  dev->version_of_latest_passed_cts = CUDA_LATEST_CONFORMANCE_VERSION;

  dev->autolocals_to_args
      = POCL_AUTOLOCALS_TO_ARGS_ONLY_IF_DYNAMIC_LOCALS_PRESENT;

  dev->has_64bit_long = 1;

  pocl_cuda_device_data_t *data = calloc (1, sizeof (pocl_cuda_device_data_t));
  dev->data = data;

  result = cuDeviceGet (&data->device, j);
  if (CUDA_CHECK_ERROR (result, "cuDeviceGet"))
    ret = CL_INVALID_DEVICE;

  /* Get specific device name */
  {
     char *name = calloc (256, sizeof (char));

     if (ret != CL_INVALID_DEVICE)
       cuDeviceGetName (name, 256, data->device);
     else
       snprintf (name, 255, "Unavailable CUDA device #%d", j);
     dev->long_name = dev->short_name = name;
  }

  /* Get other device properties */
  if (ret != CL_INVALID_DEVICE)
    {
      /* CUDA device attributes (as fetched by cuDeviceGetAttribute) are always (unsigned)
       * integers, where the OpenCL counterparts are of a variety of (other) integer types.
       * Fetch the values in an unsigned int and copy it over.
       * We also OR all return values of cuDeviceGetAttribute, and at the end we will check
       * if it's not CL_SUCCESS. We miss the exact line that failed this way, but it's
       * faster than checking after each attribute fetch.
       */
      int value = 0;
#define GET_CU_PROP(key, target) do { \
  result |= cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_##key, data->device); \
  target = value; \
} while (0)

      GET_CU_PROP (MAX_THREADS_PER_BLOCK, dev->max_work_group_size);
      GET_CU_PROP (MAX_BLOCK_DIM_X, dev->max_work_item_sizes[0]);
      GET_CU_PROP (MAX_BLOCK_DIM_Y, dev->max_work_item_sizes[1]);
      GET_CU_PROP (MAX_BLOCK_DIM_Z, dev->max_work_item_sizes[2]);
      GET_CU_PROP (MAX_SHARED_MEMORY_PER_BLOCK, dev->local_mem_size);
      GET_CU_PROP (MULTIPROCESSOR_COUNT, dev->max_compute_units);
      GET_CU_PROP (ECC_ENABLED, dev->error_correction_support);
      GET_CU_PROP (INTEGRATED, dev->host_unified_memory);
      GET_CU_PROP (TOTAL_CONSTANT_MEMORY, dev->max_constant_buffer_size);
      GET_CU_PROP (CLOCK_RATE, dev->max_clock_frequency);
      dev->max_clock_frequency /= 1000;
      GET_CU_PROP (TEXTURE_ALIGNMENT, dev->mem_base_addr_align);
      GET_CU_PROP (INTEGRATED, dev->host_unified_memory);
      data->supports_managed_memory = 0;
      GET_CU_PROP (MANAGED_MEMORY, data->supports_managed_memory);
      GET_CU_PROP (WARP_SIZE, data->warp_size);
    }
  if (CUDA_CHECK_ERROR (result, "cuDeviceGetAttribute"))
    ret = CL_INVALID_DEVICE;

  int driver_version = 0;
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuDriverGetVersion(&driver_version);
      if (CUDA_CHECK_ERROR (result, "cuDriverGetVersion"))
	{
          ret = CL_INVALID_DEVICE;
	}
      else
	{
#if CUDA_VERSION >= 11010
          if (driver_version >= 11010)
            {
              int value;
              result = cuDeviceGetAttribute (&value,
                CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED, data->device);
              /* Direct kernel access to externally registered host memory is
                 opt-in. Device storage plus versioned migrations avoids
                 exposing host allocation lifetimes to asynchronous kernels. */
              data->supports_cu_mem_host_register
                  = value && pocl_get_bool_option (
                                 "POCL_CUDA_SUPPORTS_CU_MEM_HOST_REGISTER", 0);
              if (CUDA_CHECK_ERROR (result, "cuDeviceGetAttribute"))
                ret = CL_INVALID_DEVICE;
            } else {
#else
            {
#endif
#if defined(__aarch64__) || defined(__arm__)
              // For cuda < 11.1, we don't know if the device supports cuMemHostRegister
              // or not. Let's assume that it doesn't in ARM devices.
              // This gives a false negative for Jetson Xavier, but it is the best we could do.
              data->supports_cu_mem_host_register = pocl_get_bool_option ("POCL_CUDA_SUPPORTS_CU_MEM_HOST_REGISTER", 0);
#else
              data->supports_cu_mem_host_register = pocl_get_bool_option ("POCL_CUDA_SUPPORTS_CU_MEM_HOST_REGISTER", 1);
#endif
            }
        }
    }

  data->available = CL_TRUE;
  dev->available = &data->available;

  dev->preferred_wg_size_multiple = data->warp_size;
  dev->preferred_vector_width_char = 1;
  dev->preferred_vector_width_short = 1;
  dev->preferred_vector_width_int = 1;
  dev->preferred_vector_width_long = 1;
  dev->preferred_vector_width_float = 1;
  dev->preferred_vector_width_double = 1;
  dev->native_vector_width_char = 1;
  dev->native_vector_width_short = 1;
  dev->native_vector_width_int = 1;
  dev->native_vector_width_long = 1;
  dev->native_vector_width_float = 1;
  dev->native_vector_width_double = 1;
#if GPIDL_CUDA_FUNCTIONAL_MODEL
  dev->preferred_vector_width_half = 0;
  dev->native_vector_width_half = 0;
#else
  dev->preferred_vector_width_half = 1;
  dev->native_vector_width_half = 1;

  dev->half_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
#endif
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;
  dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

#if !GPIDL_CUDA_FUNCTIONAL_MODEL
  dev->dot_product_caps
      = CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR
        | CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR;
  memset (&dev->dot_product_accel_props_8bit, 0,
          sizeof (dev->dot_product_accel_props_8bit));
  memset (&dev->dot_product_accel_props_4x8bit, 0,
          sizeof (dev->dot_product_accel_props_4x8bit));

  dev->kernel_clock_caps = CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR
                           | CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR
                           | CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR;
#endif

  dev->local_mem_type = CL_LOCAL;

  if (ret != CL_INVALID_DEVICE)
    if (choose_sm_ptx_version_combo (driver_version, dev, data) != 0)
      ret = CL_INVALID_DEVICE;
  if (ret != CL_INVALID_DEVICE)
    POCL_MSG_PRINT_INFO ("[CUDA] codegen target: sm_%d, PTX v%d.%d\n",
                         data->sm, data->ptx / 10, data->ptx % 10);

  if (ret != CL_INVALID_DEVICE)
    if (setup_kernellib (dev, data) != 0)
      ret = CL_INVALID_DEVICE;

  /* Find libdevice library */
  if (findLibDevice (data->libdevice, dev->llvm_cpu))
    {
      if (ret != CL_INVALID_DEVICE)
        {
          POCL_MSG_ERR ("[CUDA] failed to find libdevice library\n");
          dev->compiler_available = dev->linker_available = 0;
        }
    }

  /* setup builtin kernels */
  if (ret != CL_INVALID_DEVICE && dev->compiler_available)
    {
      dev->num_builtin_kernels = CUDA_BUILTIN_KERNELS;
      dev->builtins_sources_path = "builtins.cl";
      if (data->sm < 70)
        {
          dev->num_builtin_kernels
              -= 2; // last two kernels require tensor cores
        }
      dev->builtin_kernel_list = (char *)malloc (1024);
      dev->builtin_kernel_list[0] = 0;
      for (unsigned i = 0; i < dev->num_builtin_kernels; ++i)
        {
          if (i > 0)
            strcat (dev->builtin_kernel_list, ";");
          strcat (dev->builtin_kernel_list, cuda_builtin_kernels[i]);
        }
      dev->num_builtin_kernels += OPENCL_BUILTIN_KERNELS;
      for (unsigned i = 0; i < OPENCL_BUILTIN_KERNELS; ++i)
        {
          strcat (dev->builtin_kernel_list, ";");
          strcat (dev->builtin_kernel_list, opencl_builtin_kernels[i]);
        }
#ifdef ENABLE_CUDNN
      dev->num_builtin_kernels += CUDNN_BUILTIN_KERNELS;
      for (unsigned i = 0; i < CUDNN_BUILTIN_KERNELS; ++i)
        {
          strcat(dev->builtin_kernel_list, ";");
          strcat(dev->builtin_kernel_list, cudnn_builtin_kernels[i]);
        }
#endif //ENABLE_CUDNN

    }

  dev->device_side_printf = 0;

  /* Create context */
  if (ret != CL_INVALID_DEVICE)
    {
#if CUDA_VERSION >= 13000
      result = cuCtxCreate (&data->context, NULL, CU_CTX_MAP_HOST, data->device);
#else
      result = cuCtxCreate (&data->context, CU_CTX_MAP_HOST, data->device);
#endif
      if (CUDA_CHECK_ERROR (result, "cuCtxCreate"))
        ret = CL_INVALID_DEVICE;
    }

  /* Create epoch event for timing info */
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuEventCreate (&data->epoch_event, CU_EVENT_DEFAULT);
      CUDA_CHECK_ERROR (result, "cuEventCreate");

      data->epoch = pocl_gettimemono_ns ();

      result = cuEventRecord (data->epoch_event, 0);
      result = cuEventSynchronize (data->epoch_event);
      if (CUDA_CHECK_ERROR (result, "cuEventSynchronize"))
        ret = CL_INVALID_DEVICE;
    }

  /* Get global memory size */
  size_t memfree = 0, memtotal = 0;
  if (ret != CL_INVALID_DEVICE)
    result = cuMemGetInfo (&memfree, &memtotal);
  dev->max_mem_alloc_size = max (memfree, 128 * 1024 * 1024);
  dev->global_mem_size = memtotal;

#if GPIDL_CUDA_FUNCTIONAL_MODEL
  dev->svm_allocation_priority = 0;
  dev->svm_caps = 0;
#else
  if (data->supports_managed_memory)
    {
      dev->svm_allocation_priority = 2;
      dev->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
    }
  else
    {
      dev->svm_allocation_priority = 0;
      dev->svm_caps = 0;
    }
#endif

#ifdef ENABLE_SPIRV
  dev->supported_spir_v_versions = "SPIR-V_1.0";
#endif

  /* OpenCL 3.0 properties */
  /* Minimum mandated capability */
#if GPIDL_CUDA_FUNCTIONAL_MODEL
  dev->atomic_memory_capabilities
      = CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                   | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                   | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->sub_group_independent_forward_progress = CL_FALSE;
  dev->max_num_sub_groups = 0;
  dev->wg_collective_func_support = CL_FALSE;
#else
  dev->atomic_memory_capabilities
      = CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
        | CL_DEVICE_ATOMIC_ORDER_SEQ_CST | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
        | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                   | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                   | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                   | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                   | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
  dev->sub_group_independent_forward_progress = CL_TRUE;
  dev->max_num_sub_groups
      = (dev->max_work_group_size + data->warp_size - 1) / data->warp_size;
  dev->wg_collective_func_support = CL_TRUE;
#endif

  // CUDA kernels have a 4352-byte parameter area on CC >= 2.0 devices, but
  // PoCL appends work_dim, three global offsets, and twelve exact-size and
  // group-partition parameters.
  dev->max_parameter_size
      = POCL_CUDA_KERNEL_PARAMETER_BYTES
        - POCL_CUDA_IMPLICIT_PARAMETER_COUNT * sizeof (cl_uint);

#if (CUDA_DEVICE_CL_VERSION_MAJOR >= 3)
  dev->features = CUDA_DEVICE_FEATURES_30;
  dev->run_program_scope_variables_pass = CL_FALSE;
  dev->generic_as_support = GPIDL_EXPERIMENTAL_CUDA_GENERIC_AS;
  dev->non_uniform_work_group_support = CL_TRUE;

#endif
  pocl_setup_opencl_c_with_version (dev);
  pocl_setup_features_with_version (dev);

  pocl_setup_extensions_with_version (dev);

  pocl_setup_builtin_kernels_with_version (dev);

  pocl_setup_ils_with_version (dev);
  pocl_setup_spirv_queries (dev);

#ifdef ENABLE_CUDNN
  CUDNN_CALL (cudnnCreate (&cudnn));
#endif

  POCL_INIT_LOCK (data->compile_lock);
  return ret;
}

cl_int
pocl_cuda_init_queue (cl_device_id device, cl_command_queue queue)
{
  pocl_cuda_device_data_t *device_data
      = (pocl_cuda_device_data_t *)device->data;
  return pocl_cuda_create_queue (queue, device_data->context);
}

char *
pocl_cuda_build_hash (cl_device_id device)
{
  char *res = calloc (1000, sizeof (char));
  snprintf (res, 1000, "CUDA-%s-v%s", device->llvm_cpu,
            CUDA_BINARY_FORMAT_VERSION);
  return res;
}

unsigned int
pocl_cuda_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);

  int probe_count = 0;
  CUresult ret = cuInit (0);
  if (ret == CUDA_SUCCESS)
    {
      ret = cuDeviceGetCount (&probe_count);
      if (ret != CUDA_SUCCESS)
        probe_count = 0;
    }

  /* If the user requested a specific number of CUDA devices,
   * pretend we only have that many, if we can. If they requested
   * more than there are, abort informing the user of the issue.
   */
  if (env_count >= 0)
    {
      if (env_count > probe_count)
        POCL_ABORT ("[CUDA] %d devices requested, but only %d are available\n",
          env_count, probe_count);
      probe_count = env_count;
    }

  return probe_count;
}

cl_int
pocl_cuda_uninit (unsigned j, cl_device_id device)
{
  pocl_cuda_device_data_t *data = device->data;

  if (*(device->available) == CL_TRUE)
    {
      cuEventDestroy (data->epoch_event);
      cuCtxDestroy (data->context);
    }

  POCL_MEM_FREE (data);
  device->data = NULL;

  char *name = (char*)device->long_name;
  POCL_MEM_FREE (name);
  device->long_name = device->short_name = NULL;

#ifdef ENABLE_CUDNN
  CUDNN_CALL (cudnnDestroy (cudnn));
#endif

  return CL_SUCCESS;
}
