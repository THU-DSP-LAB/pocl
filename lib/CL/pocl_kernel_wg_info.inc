/* Kernel-work-group query resolution, typed values and local backend dispatch. */
typedef struct
{
  size_t capacity;
  void *value;
  size_t *size_ret;
} pocl_kernel_wg_output;

typedef cl_int (*pocl_kernel_wg_device_max_fn) (
    cl_device_id, const pocl_kernel_wg_output *);

typedef struct
{
  cl_kernel kernel;
  cl_device_id device;
  cl_kernel_work_group_info property;
  const pocl_kernel_wg_output *output;
  pocl_kernel_wg_device_max_fn device_maximum;
} pocl_kernel_wg_request;

typedef struct
{
  const pocl_kernel_wg_request *request;
  cl_device_id device;
  unsigned device_i;
} pocl_kernel_wg_context;

static cl_int
pocl_kernel_wg_error (cl_int status, const char *name, const char *reason)
{
  POCL_MSG_ERR ("%s %s\n", name, reason);
  return status;
}

static cl_int
pocl_kernel_wg_output_too_small (const pocl_kernel_wg_output *output,
                                size_t required)
{
  POCL_MSG_ERR2 ("CL_INVALID_VALUE",
                 "param_value_size (%zu) smaller than actual size (%zu)\n",
                 output->capacity, required);
  return CL_INVALID_VALUE;
}

static cl_int
pocl_kernel_wg_write (const pocl_kernel_wg_output *output, const void *value,
                      size_t bytes)
{
  if (output->value && output->capacity < bytes)
    return pocl_kernel_wg_output_too_small (output, bytes);
  if (output->value)
    memcpy (output->value, value, bytes);
  if (output->size_ret)
    *output->size_ret = bytes;
  return CL_SUCCESS;
}

static cl_int
pocl_kernel_wg_choose_device (const pocl_kernel_wg_request *request,
                              cl_device_id *result)
{
  cl_kernel kernel = request->kernel;
  if (request->device)
    {
      if (!pocl_device_is_associated_with_kernel (request->device, kernel))
        return pocl_kernel_wg_error (
            CL_INVALID_DEVICE, "CL_INVALID_DEVICE",
            "could not find the device supplied in argument");
      *result = request->device;
      return CL_SUCCESS;
    }
  if (kernel->context->num_devices > 1)
    return pocl_kernel_wg_error (CL_INVALID_DEVICE, "CL_INVALID_DEVICE",
                                 "No device given and context has > 1 device");
  *result = kernel->context->devices[0];
  return CL_SUCCESS;
}

static unsigned
pocl_kernel_wg_device_index (cl_kernel kernel, cl_device_id device)
{
  cl_device_id real_device = pocl_real_dev (device);
  unsigned result = CL_UINT_MAX;
  for (unsigned i = 0; i < kernel->program->num_devices; ++i)
    if (kernel->program->devices[i] == real_device)
      result = i;
  return result;
}

static cl_int
pocl_kernel_wg_global (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  if (meta->builtin_kernel_id == 0 || context->device->type != CL_DEVICE_TYPE_CUSTOM)
    return pocl_kernel_wg_error (
        CL_INVALID_VALUE, "CL_INVALID_VALUE",
        "only valid for custom devices or builtin kernels");
  return pocl_kernel_wg_write (context->request->output,
                               &meta->builtin_max_global_work, sizeof (size_t_3));
}

static cl_int
pocl_kernel_wg_maximum (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  if (meta->max_workgroup_size && meta->max_workgroup_size[context->device_i])
    return pocl_kernel_wg_write (
        context->request->output, &meta->max_workgroup_size[context->device_i],
        sizeof (size_t));
  return context->request->device_maximum (context->device, context->request->output);
}

static void
pocl_kernel_wg_log_compiled (const pocl_kernel_metadata_t *meta)
{
  POCL_MSG_PRINT_GENERAL ("### reqd wg sizes %zu %zu %zu\n",
                          meta->reqd_wg_size[0], meta->reqd_wg_size[1],
                          meta->reqd_wg_size[2]);
}

static cl_int
pocl_kernel_wg_compiled (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  pocl_kernel_wg_log_compiled (meta);
  return pocl_kernel_wg_write (context->request->output, meta->reqd_wg_size,
                               sizeof (size_t_3));
}

static cl_int
pocl_kernel_wg_preferred (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  size_t value = meta->preferred_wg_multiple
                     ? meta->preferred_wg_multiple[context->device_i]
                     : context->device->preferred_wg_size_multiple;
  return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
}

static cl_ulong
pocl_kernel_wg_generic_local (cl_kernel kernel, unsigned device_i)
{
  cl_ulong total = 0;
  if (kernel->meta->local_mem_size)
    total = kernel->meta->local_mem_size[device_i];
  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      const struct pocl_argument_info *info = &kernel->meta->arg_info[i];
      if (info->type == POCL_ARG_TYPE_POINTER
          && info->address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
        total += kernel->dyn_arguments[i].size;
    }
  return total;
}

static cl_int
pocl_kernel_wg_read_local (const pocl_kernel_wg_context *context, cl_ulong *value)
{
  cl_kernel kernel = context->request->kernel;
  if (context->device->ops->get_kernel_local_mem_size
      && kernel->meta->builtin_kernel_id == 0)
    return context->device->ops->get_kernel_local_mem_size (
        kernel, context->device_i, value);
  *value = pocl_kernel_wg_generic_local (kernel, context->device_i);
  return CL_SUCCESS;
}

static cl_int
pocl_kernel_wg_local (const pocl_kernel_wg_context *context)
{
  cl_ulong value = 0;
  cl_int status = pocl_kernel_wg_read_local (context, &value);
  if (status != CL_SUCCESS)
    return status;
  return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
}

static cl_int
pocl_kernel_wg_private (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  if (meta->private_mem_size)
    {
      size_t value = meta->private_mem_size[context->device_i];
      return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
    }
  const cl_ulong value = 0;
  return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
}

static cl_int
pocl_kernel_wg_spill (const pocl_kernel_wg_context *context)
{
  const pocl_kernel_metadata_t *meta = context->request->kernel->meta;
  if (meta->spill_mem_size)
    {
      size_t value = meta->spill_mem_size[context->device_i];
      return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
    }
  const cl_ulong value = 0;
  return pocl_kernel_wg_write (context->request->output, &value, sizeof (value));
}

static cl_int
pocl_kernel_wg_dispatch (const pocl_kernel_wg_context *context)
{
  switch (context->request->property)
    {
    case CL_KERNEL_GLOBAL_WORK_SIZE: return pocl_kernel_wg_global (context);
    case CL_KERNEL_WORK_GROUP_SIZE: return pocl_kernel_wg_maximum (context);
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE: return pocl_kernel_wg_compiled (context);
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: return pocl_kernel_wg_preferred (context);
    case CL_KERNEL_LOCAL_MEM_SIZE: return pocl_kernel_wg_local (context);
    case CL_KERNEL_PRIVATE_MEM_SIZE: return pocl_kernel_wg_private (context);
    case CL_KERNEL_SPILL_MEM_SIZE_INTEL: return pocl_kernel_wg_spill (context);
    default: return CL_INVALID_VALUE;
    }
}

static cl_int
pocl_kernel_wg_query (const pocl_kernel_wg_request *request)
{
  if (!IS_CL_OBJECT_VALID (request->kernel))
    return pocl_kernel_wg_error (CL_INVALID_KERNEL, "CL_INVALID_KERNEL",
                                 "(!IS_CL_OBJECT_VALID (kernel))");
  cl_device_id device = NULL;
  cl_int status = pocl_kernel_wg_choose_device (request, &device);
  if (status != CL_SUCCESS)
    return status;
  unsigned device_i = pocl_kernel_wg_device_index (request->kernel, device);
  if (device_i == CL_UINT_MAX)
    return pocl_kernel_wg_error (CL_INVALID_KERNEL, "CL_INVALID_KERNEL",
                                 "the kernel was not built for this device");
  if (*(device->available) == CL_FALSE)
    return pocl_kernel_wg_error (CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE",
                                 "device is not available");
  const pocl_kernel_wg_context context = { request, device, device_i };
  return pocl_kernel_wg_dispatch (&context);
}
