/* OpenCL runtime library: clReleaseMemObject()

   Copyright (c) 2011 Universidad Rey Juan Carlos
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
#include "pocl_cl.h"
#include "pocl_util.h"
#include "utlist.h"

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#endif

static void free_sub_buffer_data (cl_mem memobj);

static void
release_implicit_sub_buffers (cl_mem memobj)
{
  cl_mem_list_item_t *sub_buffer = NULL;
  cl_mem_list_item_t *temporary = NULL;

  LL_FOREACH_SAFE (memobj->implicit_sub_buffers, sub_buffer, temporary)
    {
      free_sub_buffer_data (sub_buffer->mem);
      if (sub_buffer->mem->last_updater != NULL)
        POname (clReleaseEvent) (sub_buffer->mem->last_updater);
      LL_DELETE (memobj->implicit_sub_buffers, sub_buffer);
      POCL_MEM_FREE (sub_buffer->mem->device_ptrs);
      POCL_MEM_FREE (sub_buffer->mem);
      POCL_MEM_FREE (sub_buffer);
    }
}

static void
release_root_mem_data (cl_mem memobj)
{
  assert (memobj->mappings == NULL);
  assert (memobj->map_count == 0);
  POCL_MSG_PRINT_REFCOUNTS (
    "Free Memory Object %" PRId64 " (%p), Flags: %" PRIu64 "\n",
    memobj->id, memobj, memobj->flags);

  pocl_release_mem_device_resources (memobj);
  release_implicit_sub_buffers (memobj);
  if (memobj->mem_host_ptr == NULL)
    return;
  if (memobj->flags & CL_MEM_USE_HOST_PTR)
    memobj->mem_host_ptr = NULL;
  else
    pocl_aligned_free (memobj->mem_host_ptr);
}

static void
unlink_content_size_buffers (cl_mem memobj)
{
  if (memobj->content_buffer != NULL)
    {
      POCL_LOCK_OBJ (memobj->content_buffer);
      assert (memobj->content_buffer->size_buffer == memobj);
      memobj->content_buffer->size_buffer = NULL;
      POCL_UNLOCK_OBJ (memobj->content_buffer);
      memobj->content_buffer = NULL;
    }

  if (memobj->size_buffer == NULL)
    return;
  POCL_LOCK_OBJ (memobj->size_buffer);
  assert (memobj->size_buffer->content_buffer == memobj);
  memobj->size_buffer->content_buffer = NULL;
  POCL_UNLOCK_OBJ (memobj->size_buffer);
  memobj->size_buffer = NULL;
}

void
pocl_release_mem_device_resources (cl_mem memobj)
{
  if (IS_IMAGE1D_BUFFER (memobj))
    return;

  for (unsigned i = 0; i < memobj->context->num_devices; ++i)
    {
      cl_device_id dev = memobj->context->devices[i];
      pocl_mem_identifier *device_ptr
        = &memobj->device_ptrs[dev->global_mem_id];
      if (device_ptr->mem_ptr == NULL)
        continue;

      if (memobj->parent != NULL)
        {
          if (dev->ops->free_subbuffer != NULL)
            dev->ops->free_subbuffer (dev, memobj);
          device_ptr->mem_ptr = NULL;
          continue;
        }

      if (*(dev->available) == CL_FALSE)
        continue;
      dev->ops->free (dev, memobj);
      device_ptr->mem_ptr = NULL;
    }
}

static cl_int
release_image1d_buffer (cl_mem memobj)
{
  cl_mem buffer = memobj->buffer;
  cl_context context = buffer->context;
  assert (buffer != NULL);
  cl_int error = POname (clReleaseMemObject) (buffer);
  assert (error == CL_SUCCESS);
  POCL_MEM_FREE (memobj->device_supports_this_image);
  error = POname (clReleaseContext) (context);
  POCL_MEM_FREE (memobj);
  return error;
}

static void
destroy_mem_object (cl_mem memobj)
{
  cl_context context = memobj->context;
  cl_event last_updater = memobj->last_updater;
  cl_mem parent = memobj->parent;

  if (parent == NULL)
    release_root_mem_data (memobj);
  else
    free_sub_buffer_data (memobj);

  POCL_MEM_FREE (memobj->device_ptrs);
  assert (memobj->destructor_callbacks == NULL);
  if (memobj->is_image)
    POCL_MEM_FREE (memobj->device_supports_this_image);
  unlink_content_size_buffers (memobj);

  if (memobj->has_device_address)
    {
      POCL_LOCK_OBJ (context);
      pocl_raw_ptr_set_erase_all_by_shadow_mem (context->raw_ptrs, memobj);
      POCL_UNLOCK_OBJ (context);
    }

  POCL_MEM_FREE (memobj->tensor_layout);
  POCL_DESTROY_OBJECT (memobj);
  POCL_MEM_FREE (memobj);
  if (parent != NULL)
    POname (clReleaseMemObject) (parent);
  POname (clReleaseContext) (context);
  if (last_updater != NULL)
    POname (clReleaseEvent) (last_updater);
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseMemObject)(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (memobj)),
                          CL_INVALID_MEM_OBJECT);

  cl_context context = memobj->context;

  POCL_LOCK_OBJ (memobj);
  POCL_RELEASE_OBJECT_UNLOCKED (memobj, new_refcount);

  if (memobj->parent != NULL)
    POCL_MSG_PRINT_REFCOUNTS (
      "Release subbuffer %" PRId64 " (%p), Refcount: %d, Parent %zu\n",
      memobj->id, memobj, new_refcount, memobj->parent->id);
  else
    POCL_MSG_PRINT_REFCOUNTS ("Release memory object %" PRId64
                              " (%p), Refcount: %d\n",
                              memobj->id, memobj, new_refcount);

  /* OpenCL 1.2 Page 118:

     After the memobj reference count becomes zero and commands queued for
     execution on a command-queue(s) that use memobj have finished, the memory
     object is deleted. If memobj is a buffer object, memobj cannot be deleted
     until all sub-buffer objects associated with memobj are deleted.
  */

  if (new_refcount != 0)
    {
      VG_REFC_NONZERO (memobj);
      POCL_UNLOCK_OBJ (memobj);
      return CL_SUCCESS;
    }

  if (memobj->destructor_callbacks != NULL)
    {
      pocl_mem_cb_push (memobj);
      POCL_UNLOCK_OBJ (memobj);
      return CL_SUCCESS;
    }
  POCL_UNLOCK_OBJ (memobj);
  VG_REFC_ZERO (memobj);

  if (memobj->is_image)
    {
      TP_FREE_IMAGE (context->id, memobj->id);
      POCL_ATOMIC_DEC (image_c);
    }
  else
    {
      TP_FREE_BUFFER (context->id, memobj->id);
      POCL_ATOMIC_DEC (buffer_c);
    }

  if (IS_IMAGE1D_BUFFER (memobj))
    return release_image1d_buffer (memobj);

  destroy_mem_object (memobj);
  return CL_SUCCESS;
}
POsym (clReleaseMemObject)

static void free_sub_buffer_data (cl_mem memobj)
{
  pocl_release_mem_device_resources (memobj);

  /* Remove the sub-buffer record from the parent buffer. */
  cl_mem_list_item_t *sub_buf;

  POCL_LOCK_OBJ_NO_CHECK (memobj->parent);

  assert (memobj->parent->sub_buffers != NULL);

  LL_SEARCH_SCALAR (memobj->parent->sub_buffers, sub_buf, mem, memobj)
    ;
  assert (sub_buf != NULL);
  assert (sub_buf->mem == memobj);

  LL_DELETE (memobj->parent->sub_buffers, sub_buf);
  free (sub_buf);

  POCL_UNLOCK_OBJ_NO_CHECK (memobj->parent);
  /* Let the parent buffer free the host pointer. */
  memobj->mem_host_ptr = NULL;
}
