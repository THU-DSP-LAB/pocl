/* Buffer migration bookkeeping uses a disjoint partition of each root.
 * User commands retain their original buffers and addresses. Internal aliases
 * may start at unaligned byte offsets: they are never public sub-buffers.
 */
#include "pocl_buffer_aliases.h"
#include "devices.h"
#include "pocl_buffer_ranges.h"
#include "pocl_util.h"
#include <stdlib.h>

static cl_mem
buffer_root (cl_mem buffer)
{
  return buffer->parent == NULL ? buffer : buffer->parent;
}

static int
contains_buffer (pocl_buffer_migration_info *list, cl_mem buffer)
{
  pocl_buffer_migration_info *entry;
  LL_FOREACH (list, entry)
  if (entry->buffer == buffer)
    return 1;
  return 0;
}

static void
free_infos (pocl_buffer_migration_info *list)
{
  pocl_buffer_migration_info *entry, *next;
  LL_FOREACH_SAFE (list, entry, next)
  free (entry);
}

void
pocl_discard_alias_migrations (pocl_buffer_migration_info *migrations)
{
  free_infos (migrations);
}

static cl_int
append_info (pocl_buffer_migration_info **list, cl_mem buffer, char read_only)
{
  pocl_buffer_migration_info *entry;
  LL_FOREACH (*list, entry)
  if (entry->buffer == buffer)
    {
      entry->read_only &= read_only;
      return CL_SUCCESS;
    }
  entry = calloc (1, sizeof (*entry));
  if (entry == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  entry->buffer = buffer;
  entry->read_only = read_only;
  DL_APPEND (*list, entry);
  return CL_SUCCESS;
}

static int
compare_roots (pocl_buffer_migration_info *a, pocl_buffer_migration_info *b)
{
  return (a->buffer->id > b->buffer->id) - (a->buffer->id < b->buffer->id);
}

pocl_buffer_migration_info *
pocl_lock_buffer_roots (pocl_buffer_migration_info *usage, cl_int *status)
{
  pocl_buffer_migration_info *roots = NULL, *entry;
  *status = CL_SUCCESS;
  LL_FOREACH (usage, entry)
  {
    cl_mem root = buffer_root (entry->buffer);
    if (root->type != CL_MEM_OBJECT_BUFFER)
      continue;
    *status = append_info (&roots, root, 1);
    if (*status != CL_SUCCESS)
      {
        free_infos (roots);
        return NULL;
      }
  }
  DL_SORT (roots, compare_roots);
  LL_FOREACH (roots, entry)
  POCL_LOCK (entry->buffer->migration_lock);
  return roots;
}

void
pocl_unlock_buffer_roots (pocl_buffer_migration_info *roots)
{
  pocl_buffer_migration_info *entry;
  LL_FOREACH (roots, entry)
  POCL_UNLOCK (entry->buffer->migration_lock);
  free_infos (roots);
}

void
pocl_release_command_buffer_locks (_cl_command_node *cmd)
{
  if (cmd->locked_buffer_roots == NULL)
    return;
  pocl_unlock_buffer_roots (cmd->locked_buffer_roots);
  cmd->locked_buffer_roots = NULL;
}

void
pocl_retain_migration_buffer_unlocked (cl_mem mem)
{
  POCL_RETAIN_OBJECT_UNLOCKED (mem);
  if (POCL_ATOMIC_LOAD (mem->implicit_sub_buffer))
    POCL_RETAIN_OBJECT (mem->parent);
}

static cl_int
initialize_device_aliases (cl_mem alias, cl_mem source)
{
  cl_mem root = alias->parent;
  for (unsigned i = 0; i < root->context->num_devices; ++i)
    {
      cl_device_id device = root->context->devices[i];
      const unsigned id = device->global_mem_id;
      pocl_mem_identifier *target = &alias->device_ptrs[id];
      const pocl_mem_identifier *original = &root->device_ptrs[id];
      if (original->mem_ptr == NULL)
        continue;
      target->mem_ptr = (char *)original->mem_ptr + alias->origin;
      target->device_addr = original->device_addr + alias->origin;
      target->extra_ptr = original->extra_ptr;
      if (device->ops->alloc_subbuffer != NULL)
        {
          cl_int status = device->ops->alloc_subbuffer (device, alias);
          if (status != CL_SUCCESS)
            return status;
        }
      target->version = source->device_ptrs[id].version;
    }
  return CL_SUCCESS;
}

static void
free_partial_alias (cl_mem alias)
{
  if (alias == NULL)
    return;
  if (alias->device_ptrs != NULL)
    pocl_release_mem_device_resources (alias);
  free (alias->device_ptrs);
  POCL_DESTROY_OBJECT (alias);
  free (alias);
}

/* The root owns internal aliases; command references keep the actual root or
 * user view alive. Retired aliases stay valid for already queued commands. */
static cl_mem
create_alias (cl_mem source, pocl_buffer_range range, cl_int *status)
{
  cl_mem root = buffer_root (source);
  cl_mem alias = calloc (1, sizeof (*alias));
  cl_mem_list_item_t *owned = calloc (1, sizeof (*owned));
  cl_mem_list_item_t *child = calloc (1, sizeof (*child));
  *status = CL_OUT_OF_HOST_MEMORY;
  if (alias == NULL || owned == NULL || child == NULL)
    {
      free (alias);
      free (owned);
      free (child);
      return NULL;
    }
  POCL_INIT_OBJECT (alias, root);
  alias->device_ptrs = calloc (POCL_ATOMIC_LOAD (pocl_num_devices),
                               sizeof (*alias->device_ptrs));
  alias->parent = root;
  alias->context = root->context;
  alias->type = CL_MEM_OBJECT_BUFFER;
  alias->size = range.size;
  alias->origin = range.origin;
  alias->flags = root->flags;
  alias->implicit_sub_buffer = POCL_ALIAS_ACTIVE;
  alias->mem_host_ptr = (char *)root->mem_host_ptr + range.origin;
  if (alias->device_ptrs == NULL)
    goto ERROR;
  POCL_LOCK_OBJ (source);
  alias->latest_version = source->latest_version;
  alias->mem_host_ptr_version = source->mem_host_ptr_version;
  alias->last_updater = source->last_updater;
  if (alias->last_updater != NULL)
    POname (clRetainEvent) (alias->last_updater);
  *status = initialize_device_aliases (alias, source);
  POCL_UNLOCK_OBJ (source);
  if (*status != CL_SUCCESS)
    goto ERROR;
  owned->mem = child->mem = alias;
  POCL_LOCK_OBJ (root);
  LL_APPEND (root->implicit_sub_buffers, owned);
  LL_APPEND (root->sub_buffers, child);
  POCL_UNLOCK_OBJ (root);
  POCL_MSG_PRINT_MEMORY (
      "Registered internal migration alias %" PRIu64 " root %" PRIu64
      " source %" PRIu64 " origin %zu size %zu\n",
      alias->id, root->id, source->id, range.origin, range.size);
  return alias;
ERROR:
  if (alias->last_updater != NULL)
    POname (clReleaseEvent) (alias->last_updater);
  free_partial_alias (alias);
  free (owned);
  free (child);
  return NULL;
}

static cl_int
initialize_partition (cl_mem root)
{
  if (root->implicit_sub_buffers != NULL)
    return CL_SUCCESS;
  POCL_LOCK_OBJ (root);
  int status = pocl_alloc_or_retain_mem_host_ptr (root);
  POCL_UNLOCK_OBJ (root);
  if (status != 0)
    return CL_OUT_OF_HOST_MEMORY;
  /* Pin the common backing store until root destruction. Individual alias
   * migration references must never free an interior host pointer. */
  cl_int error;
  if (create_alias (root, (pocl_buffer_range){ 0, root->size }, &error)
      == NULL)
    {
      POCL_LOCK_OBJ (root);
      pocl_release_mem_host_ptr (root);
      POCL_UNLOCK_OBJ (root);
    }
  return error;
}

static cl_int
split_partition (cl_mem root, size_t offset)
{
  cl_mem_list_item_t *item;
  LL_FOREACH (root->implicit_sub_buffers, item)
  {
    cl_mem source = item->mem;
    size_t end = source->origin + source->size;
    if (source->implicit_sub_buffer != POCL_ALIAS_ACTIVE
        || offset <= source->origin || offset >= end)
      continue;
    cl_int error;
    cl_mem left = create_alias (
        source, (pocl_buffer_range){ source->origin, offset - source->origin },
        &error);
    if (left == NULL)
      return error;
    cl_mem right = create_alias (
        source, (pocl_buffer_range){ offset, end - offset }, &error);
    if (right == NULL)
      {
        POCL_ATOMIC_STORE (left->implicit_sub_buffer, POCL_ALIAS_RETIRED);
        POCL_MSG_PRINT_MEMORY ("Abandoned internal migration alias %" PRIu64
                               " root %" PRIu64 " source %" PRIu64
                               " split error %d\n",
                               left->id, root->id, source->id, error);
        return error;
      }
    POCL_ATOMIC_STORE (source->implicit_sub_buffer, POCL_ALIAS_RETIRED);
    POCL_MSG_PRINT_MEMORY ("Committed internal migration split root %" PRIu64
                           " retired %" PRIu64 " left %" PRIu64
                           " right %" PRIu64 " offset %zu\n",
                           root->id, source->id, left->id, right->id, offset);
    return CL_SUCCESS;
  }
  return CL_SUCCESS;
}

static cl_int
prepare_alias_partition (pocl_buffer_migration_info *usage)
{
  cl_mem root = buffer_root (usage->buffer);
  if (root->type != CL_MEM_OBJECT_BUFFER)
    return CL_SUCCESS;
  POCL_LOCK_OBJ (root);
  int has_views = root->sub_buffers != NULL;
  POCL_UNLOCK_OBJ (root);
  if (!has_views && root->implicit_sub_buffers == NULL)
    return CL_SUCCESS;
  size_t origin = usage->buffer->parent == NULL ? 0 : usage->buffer->origin;
  size_t end = origin + usage->buffer->size;
  cl_int error = initialize_partition (root);
  if (error == CL_SUCCESS)
    error = split_partition (root, origin);
  if (error == CL_SUCCESS)
    error = split_partition (root, end);
  return error;
}

static cl_int
append_alias_migrations (pocl_buffer_migration_info **migrations,
                         pocl_buffer_migration_info *usage)
{
  cl_mem root = buffer_root (usage->buffer);
  if (root->implicit_sub_buffers == NULL)
    return append_info (migrations, usage->buffer, usage->read_only);
  size_t origin = usage->buffer->parent == NULL ? 0 : usage->buffer->origin;
  size_t end = origin + usage->buffer->size;
  cl_mem_list_item_t *item;
  LL_FOREACH (root->implicit_sub_buffers, item)
  {
    cl_mem alias = item->mem;
    if (alias->implicit_sub_buffer != POCL_ALIAS_ACTIVE
        || alias->origin < origin || alias->origin >= end)
      continue;
    cl_int error = append_info (migrations, alias, usage->read_only);
    if (error != CL_SUCCESS)
      return error;
  }
  return CL_SUCCESS;
}

pocl_buffer_migration_info *
pocl_buffer_alias_migrations (pocl_buffer_migration_info *usage,
                              cl_int *status)
{
  pocl_buffer_migration_info *migrations = NULL, *entry;
  *status = CL_SUCCESS;
  LL_FOREACH (usage, entry)
  {
    *status = prepare_alias_partition (entry);
    if (*status != CL_SUCCESS)
      return NULL;
  }
  LL_FOREACH (usage, entry)
  {
    *status = append_alias_migrations (&migrations, entry);
    if (*status != CL_SUCCESS)
      {
        free_infos (migrations);
        return NULL;
      }
  }
  return migrations;
}

void
pocl_retain_actual_buffers (pocl_buffer_migration_info *actual,
                            pocl_buffer_migration_info *migrations)
{
  pocl_buffer_migration_info *entry;
  LL_FOREACH (actual, entry)
  if (!contains_buffer (migrations, entry->buffer))
    POname (clRetainMemObject) (entry->buffer);
}

pocl_buffer_migration_info *
pocl_merge_actual_buffers (pocl_buffer_migration_info *actual,
                           pocl_buffer_migration_info *migrations)
{
  pocl_buffer_migration_info *entry, *next;
  LL_FOREACH_SAFE (actual, entry, next)
  {
    if (contains_buffer (migrations, entry->buffer))
      free (entry);
    else
      {
        entry->next = entry->prev = NULL;
        DL_APPEND (migrations, entry);
      }
  }
  return migrations;
}
