/* One dynamic shared-memory layout for CUDA launch and resource queries. */
#ifndef POCL_CUDA_LOCAL_LAYOUT_H
#define POCL_CUDA_LOCAL_LAYOUT_H
#include <limits.h>

typedef struct
{
  const pocl_kernel_metadata_t *meta;
  const pocl_argument *arguments;
  const size_t *alignments;
  size_t alignment_count;
} pocl_cuda_local_layout_input;

typedef struct
{
  unsigned bytes;
  unsigned offset;
} pocl_cuda_local_cursor;

static int
pocl_cuda_is_local_argument (const struct pocl_argument_info *info)
{
  return info->type == POCL_ARG_TYPE_POINTER
      && info->address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL;
}

static int
pocl_cuda_has_local_arguments (const pocl_kernel_metadata_t *meta)
{
  if (!meta->arg_info)
    return 0;
  for (unsigned i = 0; i < meta->num_args; ++i)
    if (pocl_cuda_is_local_argument (&meta->arg_info[i]))
      return 1;
  return 0;
}

static cl_int
pocl_cuda_append_local (pocl_cuda_local_cursor *cursor, size_t size,
                        size_t alignment)
{
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return CL_INVALID_KERNEL;
  size_t padding = (alignment - cursor->bytes % alignment) % alignment;
  if (padding > UINT_MAX - cursor->bytes
      || size > UINT_MAX - cursor->bytes - padding)
    return CL_OUT_OF_RESOURCES;
  cursor->offset = cursor->bytes + padding;
  cursor->bytes = cursor->offset + size;
  return CL_SUCCESS;
}

static cl_int
pocl_cuda_validate_local_layout (const pocl_cuda_local_layout_input *input)
{
  if (!input || !input->meta)
    return CL_INVALID_KERNEL;
  const pocl_kernel_metadata_t *meta = input->meta;
  if (meta->num_args && (!meta->arg_info || !input->arguments))
    return CL_INVALID_KERNEL;
  if (pocl_cuda_has_local_arguments (meta)
      && meta->num_locals && !meta->local_sizes)
    return CL_INVALID_KERNEL;
  return CL_SUCCESS;
}

static size_t
pocl_cuda_local_abi_count (const pocl_kernel_metadata_t *meta)
{
  return (size_t)meta->num_args
      + (pocl_cuda_has_local_arguments (meta) ? meta->num_locals : 0);
}

static int
pocl_cuda_valid_alignment_count (const pocl_cuda_local_layout_input *input)
{
  size_t count = pocl_cuda_local_abi_count (input->meta);
  return input->alignment_count == count && (!count || input->alignments);
}

static cl_int
pocl_cuda_plan_local_layout (const pocl_cuda_local_layout_input *input,
                             unsigned *offsets, unsigned *bytes)
{
  cl_int validation = pocl_cuda_validate_local_layout (input);
  if (validation != CL_SUCCESS || !bytes)
    return CL_INVALID_KERNEL;
  if (!pocl_cuda_valid_alignment_count (input))
    return CL_INVALID_KERNEL;
  const pocl_kernel_metadata_t *meta = input->meta;
  size_t count = pocl_cuda_local_abi_count (meta);
  pocl_cuda_local_cursor cursor = { 0, 0 };
  for (size_t i = 0; i < count; ++i)
    {
      int automatic = i >= meta->num_args;
      if (!automatic && !pocl_cuda_is_local_argument (&meta->arg_info[i]))
        continue;
      size_t size = automatic ? meta->local_sizes[i - meta->num_args]
                              : input->arguments[i].size;
      cl_int error = pocl_cuda_append_local (&cursor, size, input->alignments[i]);
      if (error != CL_SUCCESS)
        return error;
      if (offsets)
        offsets[i] = cursor.offset;
    }
  *bytes = cursor.bytes;
  return CL_SUCCESS;
}
#endif
