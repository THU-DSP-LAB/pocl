/* OpenCL runtime library: clCreateSubBuffer()

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "devices.h"
#include "pocl_buffer_ranges.h"
#include "pocl_cl.h"
#include "pocl_tensor_util.h"
#include "pocl_util.h"
#include "utlist.h"

typedef struct
{
  cl_mem_flags flags;
  cl_buffer_create_type type;
  const void *info;
} subbuffer_request;

typedef struct
{
  cl_int status;
  cl_mem_flags flags;
  cl_buffer_region region;
  const cl_tensor_view_exp *tensor;
} subbuffer_description;

static cl_int
validate_subbuffer_parent (cl_mem parent, const subbuffer_request *request)
{
  POCL_RETURN_ERROR_COND (!IS_CL_OBJECT_VALID (parent), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_ON (parent->is_image || IS_IMAGE1D_BUFFER (parent),
                        CL_INVALID_MEM_OBJECT,
                        "subbuffers on images not supported\n");
  POCL_RETURN_ERROR_ON (parent->parent != NULL, CL_INVALID_MEM_OBJECT,
                        "parent is already a sub-buffer\n");
  POCL_RETURN_ERROR_COND (request->info == NULL, CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND (request->type != CL_BUFFER_CREATE_TYPE_REGION
                              && request->type
                                     != CL_BUFFER_CREATE_TYPE_TENSOR_VIEW_EXP,
                          CL_INVALID_VALUE);
  return CL_SUCCESS;
}

static cl_int
validate_subbuffer_device_flags (cl_mem_flags parent, cl_mem_flags flags)
{
  POCL_RETURN_ERROR_ON (
      parent & CL_MEM_WRITE_ONLY
          && flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY),
      CL_INVALID_VALUE,
      "sub-buffer access conflicts with write-only parent\n");
  POCL_RETURN_ERROR_ON (parent & CL_MEM_READ_ONLY
                            && flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY),
                        CL_INVALID_VALUE,
                        "sub-buffer access conflicts with read-only parent\n");
  POCL_RETURN_ERROR_ON (
      flags
          & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR
             | CL_MEM_COPY_HOST_PTR),
      CL_INVALID_VALUE,
      "sub-buffer cannot specify host-pointer allocation flags\n");
  return CL_SUCCESS;
}

static cl_int
validate_subbuffer_host_flags (cl_mem_flags parent, cl_mem_flags flags)
{
  POCL_RETURN_ERROR_ON (
      parent & CL_MEM_HOST_WRITE_ONLY && flags & CL_MEM_HOST_READ_ONLY,
      CL_INVALID_VALUE, "sub-buffer host access conflicts with parent\n");
  POCL_RETURN_ERROR_ON (
      parent & CL_MEM_HOST_READ_ONLY && flags & CL_MEM_HOST_WRITE_ONLY,
      CL_INVALID_VALUE, "sub-buffer host access conflicts with parent\n");
  POCL_RETURN_ERROR_ON (
      parent & CL_MEM_HOST_NO_ACCESS
          && flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY),
      CL_INVALID_VALUE,
      "sub-buffer host access conflicts with inaccessible parent\n");
  return CL_SUCCESS;
}

static cl_int
validate_subbuffer_region (cl_mem parent, cl_buffer_region region)
{
  POCL_RETURN_ERROR_ON (region.size == 0 && !parent->has_device_address,
                        CL_INVALID_BUFFER_SIZE,
                        "buffer_create_info->size == 0\n");
  POCL_RETURN_ERROR_ON (
      !pocl_buffer_range_valid (
          (pocl_buffer_range){ region.origin, region.size }, parent->size),
      CL_INVALID_VALUE, "buffer_create_info region exceeds buffer size\n");
  POCL_RETURN_ERROR_ON (
      !parent->has_device_address
          && region.origin % parent->context->min_buffer_alignment != 0,
      CL_MISALIGNED_SUB_BUFFER_OFFSET,
      "sub-buffer origin is not device aligned\n");
  return CL_SUCCESS;
}

static subbuffer_description
parse_subbuffer_description (cl_mem parent, const subbuffer_request *request)
{
  subbuffer_description result
      = { .status = CL_SUCCESS, .flags = request->flags };
  if (request->type == CL_BUFFER_CREATE_TYPE_TENSOR_VIEW_EXP)
    {
      result.tensor = request->info;
      if (pocl_check_tensor_desc (result.tensor->tensor_desc))
        result.status = CL_INVALID_PROPERTY;
      else if (!result.tensor->tensor_desc->layout)
        result.status = CL_INVALID_VALUE;
      else
        result.region = (cl_buffer_region){ result.tensor->origin,
                                            pocl_tensor_data_size (
                                                result.tensor->tensor_desc) };
    }
  else
    result.region = *(const cl_buffer_region *)request->info;
  if (result.status == CL_SUCCESS)
    result.status = validate_subbuffer_region (parent, result.region);
  if (result.status == CL_SUCCESS)
    result.status
        = validate_subbuffer_device_flags (parent->flags, request->flags);
  if (result.status == CL_SUCCESS)
    result.status
        = validate_subbuffer_host_flags (parent->flags, request->flags);
  return result;
}

static void
inherit_subbuffer_device_version (cl_mem mem)
{
  cl_mem parent = mem->parent;
  /* Initially the sub-buffer's latest contents are in the same memory where
     the parent's latest contents are due to aliasing. */
  for (unsigned i = 0; i < mem->context->num_devices; ++i)
    {
      cl_device_id d = mem->context->devices[i];
      if (parent->device_ptrs[d->global_mem_id].version == mem->latest_version)
        {
          mem->device_ptrs[d->global_mem_id].version = mem->latest_version;
          break;
        }
    }
}

static cl_int
initialize_subbuffer_device_aliases (cl_mem mem)
{
  cl_mem parent = mem->parent;
  const cl_buffer_region region = { mem->origin, mem->size };
  const cl_buffer_region *info = &region;
  for (unsigned i = 0; i < parent->context->num_devices; ++i)
    {
      cl_device_id dev = parent->context->devices[i];
      if (parent->device_ptrs[dev->global_mem_id].mem_ptr == NULL)
        {
          /* Due to the lazy parent allocation we might not have allocated
             the parent parent yet. Let's do so now to get addresses to the
             sub-parents generated. */
          int err
              = dev->ops->alloc_mem_obj (dev, parent, parent->mem_host_ptr);
          if (err != CL_SUCCESS)
            return err;
          if (parent->device_ptrs[dev->global_mem_id].mem_ptr == NULL)
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      mem->device_ptrs[dev->global_mem_id].mem_ptr
          = (char *)parent->device_ptrs[dev->global_mem_id].mem_ptr
            + info->origin;

      /* Allocate/register the sub-parent with such devices which need to do
         something device-specific with them. */
      if (dev->ops->alloc_subbuffer != NULL)
        {
          int ret_val = dev->ops->alloc_subbuffer (dev, mem);
          if (ret_val != CL_SUCCESS)
            return ret_val;
        }

      if (mem->has_device_address)
        mem->device_ptrs[dev->global_mem_id].device_addr
            = (char *)parent->device_ptrs[dev->global_mem_id].device_addr
              + info->origin;
    }
  return CL_SUCCESS;
}

static void
destroy_partial_subbuffer (cl_mem mem)
{
  cl_mem parent = mem == NULL ? NULL : mem->parent;
  if (mem != NULL && mem->device_ptrs)
    {
      for (unsigned i = 0; i < parent->context->num_devices; ++i)
        {
          cl_device_id dev = parent->context->devices[i];
          pocl_mem_identifier *p = &mem->device_ptrs[dev->global_mem_id];
          if (p->mem_ptr != NULL && dev->ops->free_subbuffer != NULL)
            dev->ops->free_subbuffer (dev, mem);
        }
      POCL_MEM_FREE (mem->device_ptrs);
    }

  if (mem != NULL)
    {
      POCL_MEM_FREE (mem->tensor_layout);
      POCL_DESTROY_OBJECT (mem);
    }
  POCL_MEM_FREE (mem);
}

static cl_int
configure_subbuffer_description (cl_mem mem,
                                 const subbuffer_description *description)
{
  cl_mem parent = mem->parent;
  const cl_tensor_view_exp *tr_info = description->tensor;
  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->size = description->region.size;
  mem->origin = description->region.origin;
  pocl_cl_mem_inherit_flags (mem, parent, description->flags);
  mem->flags |= parent->flags & CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT;
  mem->has_device_address = parent->has_device_address;
  /* All other struct members are NULL (not valid). */

  if (tr_info)
    {
      mem->num_properties = 1;
      mem->properties[0] = CL_MEM_TENSOR_EXP;
      if (pocl_copy_tensor_desc2mem (mem, tr_info->tensor_desc))
        return CL_OUT_OF_HOST_MEMORY;
    }

  return CL_SUCCESS;
}

static cl_mem
allocate_subbuffer (cl_mem parent, const subbuffer_description *description,
                    cl_int *status)
{
  *status = CL_OUT_OF_HOST_MEMORY;
  cl_mem mem = calloc (1, sizeof (*mem));
  if (mem == NULL)
    return NULL;
  POCL_INIT_OBJECT (mem, parent);
  mem->device_ptrs = calloc (POCL_ATOMIC_LOAD (pocl_num_devices),
                             sizeof (*mem->device_ptrs));
  if (mem->device_ptrs == NULL)
    {
      destroy_partial_subbuffer (mem);
      return NULL;
    }
  mem->context = parent->context;
  mem->parent = parent;
  POCL_LOCK_OBJ (parent);
  mem->latest_version = parent->latest_version;
  mem->mem_host_ptr_version = parent->mem_host_ptr_version;
  inherit_subbuffer_device_version (mem);
  POCL_UNLOCK_OBJ (parent);
  *status = configure_subbuffer_description (mem, description);
  if (*status != CL_SUCCESS)
    {
      destroy_partial_subbuffer (mem);
      return NULL;
    }
  return mem;
}

static cl_mem
create_subbuffer_object (cl_mem parent,
                         const subbuffer_description *description,
                         cl_int *errcode_ret)
{
  cl_mem mem = NULL;
  int errcode, parent_locked = 0, references_retained = 0;
  cl_mem_list_item_t *sub_buf = NULL;
  const cl_buffer_region *info = &description->region;
  mem = allocate_subbuffer (parent, description, &errcode);
  if (mem == NULL)
    goto ERROR;

  sub_buf = calloc (1, sizeof (*sub_buf));
  POCL_GOTO_ERROR_COND ((sub_buf == NULL), CL_OUT_OF_HOST_MEMORY);
  sub_buf->mem = mem;
  POCL_RETAIN_OBJECT (mem->parent);
  POCL_RETAIN_OBJECT (mem->context);
  references_retained = 1;

  POCL_LOCK_OBJ (parent);
  parent_locked = 1;

  LL_APPEND (parent->sub_buffers, sub_buf);

  errcode = initialize_subbuffer_device_aliases (mem);
  if (errcode != CL_SUCCESS)
    goto ERROR;
  if (parent->mem_host_ptr != NULL)
    mem->mem_host_ptr = (char *)parent->mem_host_ptr + info->origin;

  POCL_UNLOCK_OBJ (parent);
  parent_locked = 0;

  POCL_MSG_PRINT_MEMORY ("Created sub-buffer %zu (%p) with size %zu, origin "
                         "%zu and parent %zu (%p)\n",
                         mem->id, mem, info->size, info->origin,
                         mem->parent->id, mem->parent);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

ERROR:
  if (parent_locked)
    {
      LL_DELETE (parent->sub_buffers, sub_buf);
      POCL_UNLOCK_OBJ (parent);
    }
  destroy_partial_subbuffer (mem);
  POCL_MEM_FREE (sub_buf);
  if (references_retained)
    {
      cl_context context = parent->context;
      POname (clReleaseMemObject) (parent);
      POname (clReleaseContext) (context);
    }
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
CL_API_ENTRY cl_mem CL_API_CALL
POname (clCreateSubBuffer) (cl_mem parent, cl_mem_flags flags,
                            cl_buffer_create_type buffer_create_type,
                            const void *buffer_create_info,
                            cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
  const subbuffer_request request
      = { flags, buffer_create_type, buffer_create_info };
  cl_int status = validate_subbuffer_parent (parent, &request);
  if (status == CL_SUCCESS)
    {
      const subbuffer_description description
          = parse_subbuffer_description (parent, &request);
      status = description.status;
      if (status == CL_SUCCESS)
        return create_subbuffer_object (parent, &description, errcode_ret);
    }
  if (errcode_ret != NULL)
    *errcode_ret = status;
  return NULL;
}
POsym (clCreateSubBuffer)
