static void
pocl_cuda_synchronize_stream (CUstream stream)
{
  CUresult result = cuStreamSynchronize (stream);
  CUDA_CHECK_ABORT (result, "cuStreamSynchronize");
}

static void *
pocl_cuda_alloc_staging (size_t size)
{
  void *staging = NULL;
  CUresult result = cuMemAllocHost (&staging, size);
  CUDA_CHECK_ABORT (result, "cuMemAllocHost");
  return staging;
}

static void
pocl_cuda_free_staging (void *staging)
{
  CUresult result = cuMemFreeHost (staging);
  CUDA_CHECK_ABORT (result, "cuMemFreeHost");
}

static void
pocl_cuda_copy_to_pageable_host (CUstream stream, void *host_ptr,
                                 CUdeviceptr device_ptr, size_t size)
{
  void *staging = pocl_cuda_alloc_staging (size);
  CUresult result = cuMemcpyDtoHAsync (staging, device_ptr, size, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoHAsync");
  pocl_cuda_synchronize_stream (stream);
  memcpy (host_ptr, staging, size);
  pocl_cuda_free_staging (staging);
}

static void
pocl_cuda_copy_from_pageable_host (CUstream stream, CUdeviceptr device_ptr,
                                   const void *host_ptr, size_t size)
{
  void *staging = pocl_cuda_alloc_staging (size);
  memcpy (staging, host_ptr, size);
  CUresult result = cuMemcpyHtoDAsync (device_ptr, staging, size, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyHtoDAsync");
  pocl_cuda_synchronize_stream (stream);
  pocl_cuda_free_staging (staging);
}

void
pocl_cuda_submit_read (CUstream stream, void *host_ptr,
                       const pocl_mem_identifier *mem_id, size_t offset,
                       size_t cb)
{
  if (mem_id->extra_ptr != NULL)
    {
      const void *source = (const char *)mem_id->extra_ptr + offset;
      if (source == host_ptr)
        return;

      pocl_cuda_synchronize_stream (stream);
      memmove (host_ptr, source, cb);
      return;
    }

  const void *source = (const char *)mem_id->mem_ptr + offset;
  if (source == host_ptr)
    return;

  POCL_MSG_PRINT_CUDA ("staged cuMemcpyDtoHAsync %p -> %p / %zu B \n",
                       source, host_ptr, cb);
  pocl_cuda_copy_to_pageable_host (stream, host_ptr, (CUdeviceptr)source, cb);
}

static CUresult
pocl_cuda_submit_repeated_pattern (CUstream stream, void *mem_ptr,
                                   size_t size_in_bytes, size_t offset,
                                   const void *pattern, size_t pattern_size)
{
  CUdeviceptr dest = (CUdeviceptr)(((char *)mem_ptr) + offset);
  pocl_cuda_copy_from_pageable_host (stream, dest, pattern, pattern_size);

  size_t copied = pattern_size;
  CUresult result = CUDA_SUCCESS;
  while (copied < size_in_bytes)
    {
      size_t remaining = size_in_bytes - copied;
      size_t chunk_size = copied < remaining ? copied : remaining;
      result = cuMemcpyDtoDAsync (dest + copied, dest, chunk_size, stream);
      if (result != CUDA_SUCCESS)
        return result;
      copied += chunk_size;
    }
  return CUDA_SUCCESS;
}

void
pocl_cuda_submit_memfill (CUstream stream, void *mem_ptr, size_t size_in_bytes,
                          size_t offset, const void *pattern,
                          size_t pattern_size)
{
  CUresult result;
  switch (pattern_size)
    {
    case 1:
      result
          = cuMemsetD8Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                             *(unsigned char *)pattern, size_in_bytes, stream);
      break;
    case 2:
      result = cuMemsetD16Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                                 *(unsigned short *)pattern, size_in_bytes / 2,
                                 stream);
      break;
    case 4:
      result = cuMemsetD32Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                                 *(unsigned int *)pattern, size_in_bytes / 4,
                                 stream);
      break;
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      result = pocl_cuda_submit_repeated_pattern (
          stream, mem_ptr, size_in_bytes, offset, pattern, pattern_size);
      break;
    default:
      POCL_ABORT ("unrecognized pattern_size");
    }
  CUDA_CHECK_ABORT (result, "cuMemset*Async");
}

void
pocl_cuda_submit_write (CUstream stream, const void *host_ptr,
                        pocl_mem_identifier *mem_id, size_t offset, size_t cb)
{
  if (mem_id->extra_ptr != NULL)
    {
      void *destination = (char *)mem_id->extra_ptr + offset;
      if (host_ptr == destination)
        return;

      pocl_cuda_synchronize_stream (stream);
      memmove (destination, host_ptr, cb);
      return;
    }

  void *destination = (char *)mem_id->mem_ptr + offset;
  if (host_ptr == destination)
    return;

  POCL_MSG_PRINT_CUDA ("staged cuMemcpyHtoDAsync %p -> %p / %zu B \n",
                       host_ptr, destination, cb);
  pocl_cuda_copy_from_pageable_host (stream, (CUdeviceptr)destination,
                                     host_ptr, cb);
}

void
pocl_cuda_submit_copy (CUstream stream,
                       const pocl_mem_identifier *__restrict__ src_mem_id,
                       size_t src_offset,
                       const pocl_mem_identifier *__restrict__ dst_mem_id,
                       size_t dst_offset, size_t cb)
{
  void *src_ptr = (char *)(src_mem_id->extra_ptr != NULL
                             ? src_mem_id->extra_ptr
                             : src_mem_id->mem_ptr)
                  + src_offset;
  void *dst_ptr = (char *)(dst_mem_id->extra_ptr != NULL
                             ? dst_mem_id->extra_ptr
                             : dst_mem_id->mem_ptr)
                  + dst_offset;

  if (src_ptr == dst_ptr)
    return;

  CUresult result;
  if (src_mem_id->extra_ptr != NULL && dst_mem_id->extra_ptr != NULL)
    {
      pocl_cuda_synchronize_stream (stream);
      memmove (dst_ptr, src_ptr, cb);
      return;
    }

  if (src_mem_id->extra_ptr != NULL)
    {
      POCL_MSG_PRINT_CUDA ("cuMemcpyHtoDAsync %p -> %p / %zu B \n", src_ptr,
                           dst_ptr, cb);
      result = cuMemcpyHtoDAsync ((CUdeviceptr)dst_ptr, src_ptr, cb, stream);
      CUDA_CHECK_ABORT (result, "cuMemcpyHtoDAsync");
      return;
    }

  if (dst_mem_id->extra_ptr != NULL)
    {
      POCL_MSG_PRINT_CUDA ("cuMemcpyDtoHAsync %p -> %p / %zu B \n", src_ptr,
                           dst_ptr, cb);
      result = cuMemcpyDtoHAsync (dst_ptr, (CUdeviceptr)src_ptr, cb, stream);
      CUDA_CHECK_ABORT (result, "cuMemcpyDtoHAsync");
      return;
    }

  POCL_MSG_PRINT_CUDA ("cuMemcpyDtoDAsync %p -> %p / %zu B \n", src_ptr,
                       dst_ptr, cb);
  result = cuMemcpyDtoDAsync ((CUdeviceptr)dst_ptr, (CUdeviceptr)src_ptr, cb,
                              stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoDAsync");
}

void
pocl_cuda_submit_copy_p2p (CUstream stream, cl_device_id src_device,
                           void *__restrict__ src_mem_ptr, size_t src_offset,
                           cl_device_id dst_device,
                           void *__restrict__ dst_mem_ptr, size_t dst_offset,
                           size_t cb)
{
  void *src_ptr = src_mem_ptr + src_offset;
  void *dst_ptr = dst_mem_ptr + dst_offset;

  CUresult result;
  POCL_MSG_PRINT_CUDA ("cuMemcpyPeerAsync %p -> %p / %zu B \n", src_ptr,
                       dst_ptr, cb);
  result = cuMemcpyPeerAsync (
      (CUdeviceptr)dst_ptr,
      ((pocl_cuda_device_data_t *)dst_device->data)->context,
      (CUdeviceptr)src_ptr,
      ((pocl_cuda_device_data_t *)dst_device->data)->context, cb, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyPeerAsync");
}

static size_t
pocl_cuda_rect_offset (const size_t *origin, size_t row_pitch,
                       size_t slice_pitch)
{
  return origin[2] * slice_pitch + origin[1] * row_pitch + origin[0];
}

typedef struct
{
  void *base;
  const size_t *origin;
  const size_t *region;
  size_t row_pitch;
  size_t slice_pitch;
} pocl_cuda_host_rect_t;

static size_t
pocl_cuda_host_rect_staging_size (const pocl_cuda_host_rect_t *rect)
{
  return rect->region[0] * rect->region[1] * rect->region[2];
}

static void
pocl_cuda_pack_host_rect (void *staging, const pocl_cuda_host_rect_t *rect)
{
  const char *source = (const char *)rect->base
                       + pocl_cuda_rect_offset (
                           rect->origin, rect->row_pitch, rect->slice_pitch);
  char *destination = (char *)staging;

  for (size_t z = 0; z < rect->region[2]; ++z)
    for (size_t y = 0; y < rect->region[1]; ++y)
      {
        size_t source_offset = z * rect->slice_pitch + y * rect->row_pitch;
        size_t destination_offset
            = (z * rect->region[1] + y) * rect->region[0];
        memcpy (destination + destination_offset, source + source_offset,
                rect->region[0]);
      }
}

static void
pocl_cuda_unpack_host_rect (const void *staging,
                            const pocl_cuda_host_rect_t *rect)
{
  const char *source = (const char *)staging;
  char *destination = (char *)rect->base
                      + pocl_cuda_rect_offset (
                          rect->origin, rect->row_pitch, rect->slice_pitch);

  for (size_t z = 0; z < rect->region[2]; ++z)
    for (size_t y = 0; y < rect->region[1]; ++y)
      {
        size_t source_offset
            = (z * rect->region[1] + y) * rect->region[0];
        size_t destination_offset
            = z * rect->slice_pitch + y * rect->row_pitch;
        memcpy (destination + destination_offset, source + source_offset,
                rect->region[0]);
      }
}

void
pocl_cuda_submit_read_rect (CUstream stream, void *__restrict__ const host_ptr,
                            void *__restrict__ const device_ptr,
                            const size_t *__restrict__ const buffer_origin,
                            const size_t *__restrict__ const host_origin,
                            const size_t *__restrict__ const region,
                            size_t const buffer_row_pitch,
                            size_t const buffer_slice_pitch,
                            size_t const host_row_pitch,
                            size_t const host_slice_pitch)
{
  CUDA_MEMCPY3D params = { 0 };

  size_t device_offset = pocl_cuda_rect_offset (
      buffer_origin, buffer_row_pitch, buffer_slice_pitch);
  pocl_cuda_host_rect_t host_rect = {
    .base = host_ptr,
    .origin = host_origin,
    .region = region,
    .row_pitch = host_row_pitch,
    .slice_pitch = host_slice_pitch,
  };
  size_t staging_size = pocl_cuda_host_rect_staging_size (&host_rect);
  void *staging = pocl_cuda_alloc_staging (staging_size);

  POCL_MSG_PRINT_CUDA ("staged cuMemcpy3DAsync / READ_RECT %p -> %p \n",
                       device_ptr, host_ptr);

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.dstMemoryType = CU_MEMORYTYPE_HOST;
  params.dstHost = staging;
  params.dstPitch = region[0];
  params.dstHeight = region[1];

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = (CUdeviceptr)device_ptr + device_offset;
  params.srcPitch = buffer_row_pitch;
  params.srcHeight = buffer_slice_pitch / buffer_row_pitch;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
  pocl_cuda_synchronize_stream (stream);
  pocl_cuda_unpack_host_rect (staging, &host_rect);
  pocl_cuda_free_staging (staging);
}

void
pocl_cuda_submit_write_rect (CUstream stream,
                             const void *__restrict__ const host_ptr,
                             void *__restrict__ const device_ptr,
                             const size_t *__restrict__ const buffer_origin,
                             const size_t *__restrict__ const host_origin,
                             const size_t *__restrict__ const region,
                             size_t const buffer_row_pitch,
                             size_t const buffer_slice_pitch,
                             size_t const host_row_pitch,
                             size_t const host_slice_pitch)
{
  CUDA_MEMCPY3D params = { 0 };

  size_t device_offset = pocl_cuda_rect_offset (
      buffer_origin, buffer_row_pitch, buffer_slice_pitch);
  pocl_cuda_host_rect_t host_rect = {
    .base = (void *)host_ptr,
    .origin = host_origin,
    .region = region,
    .row_pitch = host_row_pitch,
    .slice_pitch = host_slice_pitch,
  };
  size_t staging_size = pocl_cuda_host_rect_staging_size (&host_rect);
  void *staging = pocl_cuda_alloc_staging (staging_size);
  pocl_cuda_pack_host_rect (staging, &host_rect);

  POCL_MSG_PRINT_CUDA ("staged cuMemcpy3DAsync / WRITE_RECT %p -> %p \n",
                       host_ptr, device_ptr);

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost = staging;
  params.srcPitch = region[0];
  params.srcHeight = region[1];

  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr)device_ptr + device_offset;
  params.dstPitch = buffer_row_pitch;
  params.dstHeight = buffer_slice_pitch / buffer_row_pitch;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
  pocl_cuda_synchronize_stream (stream);
  pocl_cuda_free_staging (staging);
}

void
pocl_cuda_submit_copy_rect (CUstream stream,
                            cl_device_id dev,
                            void* src_ptr,
                            void* dst_ptr,
                            const size_t *__restrict__ const src_origin,
                            const size_t *__restrict__ const dst_origin,
                            const size_t *__restrict__ const region,
                            size_t const src_row_pitch,
                            size_t const src_slice_pitch,
                            size_t const dst_row_pitch,
                            size_t const dst_slice_pitch)
{
  CUDA_MEMCPY3D params = { 0 };

  size_t src_offset
      = pocl_cuda_rect_offset (src_origin, src_row_pitch, src_slice_pitch);
  size_t dst_offset
      = pocl_cuda_rect_offset (dst_origin, dst_row_pitch, dst_slice_pitch);

  POCL_MSG_PRINT_CUDA ("cuMemcpy3D / COPY_RECT %p -> %p \n", src_ptr, dst_ptr);

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcDevice = (CUdeviceptr)src_ptr + src_offset;
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstDevice = (CUdeviceptr)dst_ptr + dst_offset;
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  params.srcMemoryType = params.dstMemoryType = CU_MEMORYTYPE_DEVICE;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
}

void
pocl_cuda_submit_map_mem (CUstream stream, pocl_mem_identifier *p,
                          size_t offset, size_t size, void *host_ptr)
{
  assert (host_ptr != NULL);

  if (p->extra_ptr != NULL)
    return;

  POCL_MSG_PRINT_CUDA ("cuMemcpyDtoHAsync %p / %zu B \n", host_ptr, size);

  void *buf_ptr = p->mem_ptr;

  pocl_cuda_copy_to_pageable_host (
      stream, host_ptr, (CUdeviceptr)(buf_ptr + offset), size);
}

void *
pocl_cuda_submit_unmap_mem (CUstream stream, pocl_mem_identifier *dst_mem_id,
                            size_t offset, size_t size, void *host_ptr,
                            cl_map_flags map_flags)
{
  /* Only copy back if mapped for writing */
  if (map_flags == CL_MAP_READ)
    return NULL;

  /* Mapped host allocations use the same storage on the host and device. */
  if (dst_mem_id->extra_ptr != NULL)
    return NULL;

  if (host_ptr)
    {
      pocl_cuda_copy_from_pageable_host (
          stream, (CUdeviceptr)(dst_mem_id->mem_ptr + offset), host_ptr, size);
    }
  return NULL;
}

// https://docs.nvidia.com/cuda/ptx-compiler-api/index.html#basic-usage

/* build a program with builtin kernels. */
