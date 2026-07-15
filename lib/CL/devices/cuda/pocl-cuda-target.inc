static void
pocl_cuda_abort_on_error (CUresult result, unsigned line, const char *func,
                          const char *code, const char *api)
{
  if (result != CUDA_SUCCESS)
    {
      const char *err_name;
      const char *err_string;
      cuGetErrorName (result, &err_name);
      cuGetErrorString (result, &err_string);
      POCL_MSG_PRINT2 (CUDA, func, line, "Error during %s\n", api);
      POCL_ABORT ("%s: %s\n", err_name, err_string);
    }
}

static int
pocl_cuda_error (CUresult result, unsigned line, const char *func,
                          const char *code, const char *api)
{
  int err = (result != CUDA_SUCCESS);
  if (err)
    {
      const char *err_name;
      const char *err_string;
      cuGetErrorName (result, &err_name);
      cuGetErrorString (result, &err_string);
      POCL_MSG_ERR ("CUDA error during %s. %s: %s\n", api, err_name, err_string);
    }
  return err;
}

#define CUDA_CHECK_ABORT(result, api)                                         \
  pocl_cuda_abort_on_error (result, __LINE__, __FUNCTION__, #result, api)

#define CUDA_CHECK_ERROR(result, api)                                         \
  pocl_cuda_error (result, __LINE__, __FUNCTION__, #result, api)

/* Parses SM version string 'str' (expecting form sm_XX) to integer in
 * form of 'MAJOR * 10 + MINOR'.  Returns 0 if parsing failed. */
static int
parse_sm_version_string (const char *str, unsigned *parsed_sm)
{
  if (str == NULL || parsed_sm == NULL || strncmp (str, "sm_", 3) != 0)
    return -1;

  char *end = NULL;
  errno = 0;
  unsigned long parsed_num = strtoul (str + 3, &end, 10);
  if (errno != 0 || end == str + 3 || *end != '\0' || parsed_num == 0
      || parsed_num > UINT_MAX)
    return -1;

  *parsed_sm = (unsigned)parsed_num;
  return 0;
}
struct cuda_ptx_version_pair_s
{
  unsigned cuda_version;
  unsigned ptx_version;
};

/* Maps CUDA driver version to PTX version the driver supports at most
 * (there doesn't seem to be a way to query supported PTX versions). */
const struct cuda_ptx_version_pair_s cuda_ptx_version_map[] = {
  /* Based on [PTX ISA v8.5 - 12. Release Notes]. For simplicity,
     minor CUDA versions are (mostly) skipped for now. */
  { 12000, 80 }, { 11000, 70 }, { 10000, 63 }, { 9000, 60 }, { 8000, 50 },
  { 7000, 42 },  { 6000, 40 },  { 5000, 31 },  { 4010, 30 }, { 4000, 23 },
  { 3000, 20 },  { 2000, 12 },  { 1000, 10 },  { 0, 0 } /* Sentinel. */
};

/* Return a PTX version range that is supported by the CUDA driver for
   the specified SM architecture. */
static int
get_valid_ptx_version_range (unsigned cuda_version, unsigned sm_version,
                             unsigned *min_ptx_out, unsigned *max_ptx_out)
{
  if (min_ptx_out == NULL || max_ptx_out == NULL)
    return -1;

  /* SM version sets lower bound. PTX version needs to be high enough
     target the given SM version. */
  unsigned min_ptx = 0;

  /* Based on [PTX ISA v8.5 - 12. Release Notes]. */
  if (sm_version >= 89)
    min_ptx = 78;
  else if (sm_version >= 87)
    min_ptx = 74;
  else if (sm_version >= 86)
    min_ptx = 71;
  else if (sm_version >= 80)
    min_ptx = 70;
  else if (sm_version >= 75)
    min_ptx = 63;
  else if (sm_version >= 72)
    min_ptx = 61;
  else if (sm_version >= 70)
    min_ptx = 60;
  else if (sm_version >= 60)
    min_ptx = 50;
  else if (sm_version >= 53)
    min_ptx = 42;
  else if (sm_version >= 52)
    min_ptx = 41;
  else if (sm_version >= 50)
    min_ptx = 40;
  else if (sm_version >= 35)
    min_ptx = 31;
  else if (sm_version >= 30)
    min_ptx = 30;
  else if (sm_version >= 20)
    min_ptx = 20;
  else if (sm_version >= 13)
    min_ptx = 12;
  else if (sm_version >= 11)
    min_ptx = 10;

  /* CUDA runtime sets upper bound. PTX version may not be higher than
     what the runtime recognizes. */

  unsigned max_ptx = 0;
  for (unsigned i = 0; cuda_ptx_version_map[i].cuda_version; i++)
    if (cuda_version >= cuda_ptx_version_map[i].cuda_version)
      {
        max_ptx = cuda_ptx_version_map[i].ptx_version;
        break;
      }

  if (min_ptx == 0 || max_ptx == 0 || min_ptx > max_ptx)
    {
      POCL_MSG_ERR ("[CUDA] unsupported SM/driver combination: sm_%u, "
                    "CUDA driver %u\n",
                    sm_version, cuda_version);
      return -1;
    }

  *min_ptx_out = min_ptx;
  *max_ptx_out = max_ptx;
  return 0;
}

static int
choose_sm_ptx_version_combo (unsigned cuda_version, cl_device_id dev,
                             pocl_cuda_device_data_t *data)
{
  /* Choose SM version to target. */

  unsigned selected_sm = 0;

  const char *user_sm = pocl_get_string_option ("POCL_CUDA_GPU_ARCH", NULL);
  if (user_sm && parse_sm_version_string (user_sm, &selected_sm) != 0)
    {
      POCL_MSG_ERR ("[CUDA] invalid POCL_CUDA_GPU_ARCH value '%s'; expected "
                    "sm_<major><minor>\n",
                    user_sm);
      return -1;
    }

  if (!selected_sm)
    {
      int sm_maj = 0, sm_min = 0;
      CUresult result = cuDeviceGetAttribute (
          &sm_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, data->device);
      result |= cuDeviceGetAttribute (
          &sm_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, data->device);
      if (CUDA_CHECK_ERROR (result, "query compute capability"))
        return -1;

      /* Existing CUDA architecture names suggest that minor versions
         are limited to a single digit. */
      if (sm_maj <= 0 || sm_min < 0 || sm_min >= 10)
        {
          POCL_MSG_ERR ("[CUDA] invalid compute capability %d.%d\n", sm_maj,
                        sm_min);
          return -1;
        }

      selected_sm = sm_maj * 10 + sm_min;
    }

  char *gpu_arch = calloc (16, sizeof (char));
  snprintf (gpu_arch, 16, "sm_%d", selected_sm);
  dev->llvm_cpu = gpu_arch;

  /* Choose PTX version to target. */

  unsigned min_ptx = 0, max_ptx = 0;
  if (get_valid_ptx_version_range (cuda_version, selected_sm, &min_ptx,
                                   &max_ptx)
      != 0)
    return -1;

  unsigned selected_ptx = max_ptx;
  if (selected_sm <= 75)
    for (unsigned candidate = 60; candidate < 64; ++candidate)
      if (candidate >= min_ptx && candidate <= max_ptx)
        {
          selected_ptx = candidate;
          break;
        }

  data->sm = selected_sm;
  data->ptx = selected_ptx;
  return 0;
}

/* Setup the kernel library for the selected target. */
static int
setup_kernellib (cl_device_id dev, pocl_cuda_device_data_t *data)
{
  assert (data && "data must not be NULL!");
  assert (data->sm && data->ptx && "CUDA target must be known!");

  int is_64bit_target = (sizeof (void *) == 8);

  if (data->ptx >= 64 && data->sm >= 70)
    {
      dev->kernellib_name = is_64bit_target ? "kernel-nvptx64-sm70-ptx64"
                                            : "kernel-nvptx-sm70-ptx64";
    }
  else if (data->ptx >= 60 && data->ptx < 64 && data->sm <= 75)
    {
      dev->kernellib_name = is_64bit_target ? "kernel-nvptx64-sm30-ptx60-63"
                                            : "kernel-nvptx-sm30-ptx60-63";
    }
  else if (data->ptx < 60 && data->sm <= 75)
    {
      dev->kernellib_name = is_64bit_target ? "kernel-nvptx64-sm30-ptx60lt"
                                            : "kernel-nvptx-sm30-ptx60lt";
    }
  else
    {
      POCL_MSG_ERR ("[CUDA] no kernel library for sm_%u and PTX %u.%u\n",
                    data->sm, data->ptx / 10, data->ptx % 10);
      return -1;
    }
  return 0;
}
