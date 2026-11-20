/* Internal query helpers only: injected backend replies, no public CL/CUDA API. */
#include "devices/devices.h"
#include "pocl_util.h"
#include <stdio.h>
#include <stdlib.h>
#include "pocl_kernel_wg_info.inc"

typedef struct { cl_int status; cl_ulong value; } Reply;
typedef struct
{
  struct _cl_kernel kernel;
  struct _cl_program program;
  struct _cl_context context;
  struct _cl_device_id device[2];
  struct pocl_device_ops ops;
  cl_device_id devices[2];
  cl_device_id requested;
  cl_bool available;
  pocl_kernel_metadata_t meta;
  struct pocl_argument_info info[2];
  pocl_argument arguments[2];
  cl_ulong local[2], private_size[2], spill[2];
  size_t maximum[2], preferred[2];
  Reply backend[2], maximum_reply;
  void *backend_data[2];
} Fixture;

static void require (int condition, const char *message)
{
  if (!condition) { fprintf (stderr, "%s\n", message); exit (1); }
}

static cl_int backend_local (cl_kernel kernel, unsigned device_i, cl_ulong *value)
{
  const Reply *reply = kernel->meta->data[device_i];
  *value = reply->value;
  return reply->status;
}

static cl_int device_maximum (cl_device_id device,
                              const pocl_kernel_wg_output *output)
{
  const Reply *reply = device->data;
  if (reply->status != CL_SUCCESS)
    return reply->status;
  size_t value = reply->value;
  return pocl_kernel_wg_write (output, &value, sizeof (value));
}

static void initialize (Fixture *fixture)
{
  memset (fixture, 0, sizeof (*fixture));
  fixture->available = CL_TRUE;
  for (unsigned i = 0; i < 2; ++i)
    {
      fixture->devices[i] = fixture->device + i;
      fixture->device[i].available = &fixture->available;
      fixture->device[i].ops = &fixture->ops;
      fixture->device[i].preferred_wg_size_multiple = 64;
      fixture->device[i].data = &fixture->maximum_reply;
      fixture->backend[i] = (Reply) { CL_SUCCESS, 700 + i };
      fixture->backend_data[i] = fixture->backend + i;
      fixture->local[i] = 100 * (i + 1);
      fixture->private_size[i] = 5 + i;
      fixture->spill[i] = 7 + i;
      fixture->maximum[i] = 128 * (i + 1);
      fixture->preferred[i] = 16 * (i + 1);
    }
  fixture->maximum_reply = (Reply) { CL_SUCCESS, 77 };
  fixture->context.num_devices = 1;
  fixture->context.devices = fixture->devices;
  fixture->program.num_devices = 1;
  fixture->program.devices = fixture->devices;
  fixture->kernel.context = &fixture->context;
  fixture->kernel.program = &fixture->program;
  fixture->kernel.meta = &fixture->meta;
  fixture->kernel.dyn_arguments = fixture->arguments;
  fixture->meta.num_args = 2;
  fixture->meta.arg_info = fixture->info;
  fixture->meta.local_mem_size = fixture->local;
  fixture->meta.private_mem_size = fixture->private_size;
  fixture->meta.spill_mem_size = fixture->spill;
  fixture->meta.max_workgroup_size = fixture->maximum;
  fixture->meta.preferred_wg_multiple = fixture->preferred;
  fixture->meta.data = fixture->backend_data;
  fixture->info[0].type = POCL_ARG_TYPE_POINTER;
  fixture->info[0].address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
  fixture->info[1].type = POCL_ARG_TYPE_POINTER;
  fixture->info[1].address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
  fixture->arguments[0].size = 91;
  fixture->arguments[1].size = 13;
  SET_VALIDITY_MARKERS (&fixture->kernel);
}

static cl_int query (Fixture *fixture, cl_kernel_work_group_info property,
                      const pocl_kernel_wg_output *output)
{
  const pocl_kernel_wg_request request = {
    &fixture->kernel, fixture->requested, property, output, device_maximum
  };
  return pocl_kernel_wg_query (&request);
}

static void expect_scalar (Fixture *fixture, cl_kernel_work_group_info property,
                            cl_ulong expected)
{
  cl_ulong value = 999;
  size_t bytes = 0;
  const pocl_kernel_wg_output output = { sizeof (value), &value, &bytes };
  require (query (fixture, property, &output) == CL_SUCCESS, "scalar query status");
  require (value == expected, "scalar query value");
  require (bytes == sizeof (value), "scalar query size");
}

static void test_scalar_properties (void)
{
  Fixture f; initialize (&f);
  expect_scalar (&f, CL_KERNEL_LOCAL_MEM_SIZE, 113);
  expect_scalar (&f, CL_KERNEL_PRIVATE_MEM_SIZE, 5);
  expect_scalar (&f, CL_KERNEL_SPILL_MEM_SIZE_INTEL, 7);
  expect_scalar (&f, CL_KERNEL_WORK_GROUP_SIZE, 128);
  expect_scalar (&f, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 16);
  f.meta.private_mem_size = NULL; f.meta.spill_mem_size = NULL;
  f.meta.preferred_wg_multiple = NULL;
  expect_scalar (&f, CL_KERNEL_PRIVATE_MEM_SIZE, 0);
  expect_scalar (&f, CL_KERNEL_SPILL_MEM_SIZE_INTEL, 0);
  expect_scalar (&f, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 64);
}

static void test_local_dispatch_and_slots (void)
{
  Fixture f; initialize (&f);
  f.ops.get_kernel_local_mem_size = backend_local;
  for (unsigned i = 0; i < 3; ++i)
    expect_scalar (&f, CL_KERNEL_LOCAL_MEM_SIZE, 700);
  require (f.local[0] == 100, "backend query mutated shared base");
  f.meta.builtin_kernel_id = 1;
  expect_scalar (&f, CL_KERNEL_LOCAL_MEM_SIZE, 113);
  f.meta.builtin_kernel_id = 0;
  f.context.num_devices = 2; f.program.num_devices = 2;
  f.requested = f.device + 1;
  expect_scalar (&f, CL_KERNEL_LOCAL_MEM_SIZE, 701);
  f.ops.get_kernel_local_mem_size = NULL;
  expect_scalar (&f, CL_KERNEL_LOCAL_MEM_SIZE, 213);
}

static void test_sizes_and_failed_writeback (void)
{
  Fixture f; initialize (&f);
  cl_ulong value = 999;
  size_t bytes = 123;
  pocl_kernel_wg_output output = { sizeof (value) - 1, &value, &bytes };
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_INVALID_VALUE,
           "undersized output accepted");
  require (value == 999, "undersized query wrote output");
  require (bytes == 123, "undersized query wrote size_ret");
  output.value = NULL;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_SUCCESS,
           "size-only query rejected small unused capacity");
  require (bytes == sizeof (cl_ulong), "size-only result");
  output.size_ret = NULL;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_SUCCESS,
           "null value and size_ret behavior changed");
  f.ops.get_kernel_local_mem_size = backend_local;
  f.backend[0].status = CL_OUT_OF_RESOURCES;
  output = (pocl_kernel_wg_output) { sizeof (value), &value, &bytes };
  bytes = 123;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_OUT_OF_RESOURCES,
           "backend failure hidden by generic path");
  require (value == 999, "backend failure published temporary value");
  require (bytes == 123, "backend failure changed size_ret");
}

static void test_array_properties (void)
{
  Fixture f; initialize (&f);
  size_t_3 value = { { 0, 0, 0 } };
  size_t bytes = 0;
  const pocl_kernel_wg_output output = { sizeof (value), &value, &bytes };
  f.meta.reqd_wg_size[0] = 4; f.meta.reqd_wg_size[1] = 8;
  f.meta.reqd_wg_size[2] = 2;
  require (query (&f, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &output) == CL_SUCCESS,
           "compile group query");
  require (value.size[0] == 4, "compile x");
  require (value.size[1] == 8, "compile y");
  require (value.size[2] == 2, "compile z");
  require (bytes == sizeof (value), "compile array bytes");
  require (query (&f, CL_KERNEL_GLOBAL_WORK_SIZE, &output) == CL_INVALID_VALUE,
           "global work allowed for non-builtin");
  f.meta.builtin_kernel_id = 1; f.device[0].type = CL_DEVICE_TYPE_CUSTOM;
  f.meta.builtin_max_global_work = (size_t_3) { { 9, 10, 11 } };
  require (query (&f, CL_KERNEL_GLOBAL_WORK_SIZE, &output) == CL_SUCCESS,
           "custom builtin global query");
  require (value.size[0] == 9, "global x");
  require (value.size[1] == 10, "global y");
  require (value.size[2] == 11, "global z");
}

static void test_resolution_errors (void)
{
  Fixture f; initialize (&f);
  cl_ulong value = 999;
  size_t bytes = 123;
  const pocl_kernel_wg_output output = { sizeof (value), &value, &bytes };
  f.context.num_devices = 2;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_INVALID_DEVICE,
           "ambiguous implicit device accepted");
  f.requested = f.device + 1;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_INVALID_KERNEL,
           "device absent from build accepted");
  f.context.num_devices = 1;
  require (query (&f, CL_KERNEL_LOCAL_MEM_SIZE, &output) == CL_INVALID_DEVICE,
           "unassociated explicit device accepted");
  f.requested = NULL; f.available = CL_FALSE;
  require (query (&f, 0xffffffffu, &output) == CL_DEVICE_NOT_AVAILABLE,
           "unavailable device precedence changed");
  f.available = CL_TRUE;
  require (query (&f, 0xffffffffu, &output) == CL_INVALID_VALUE,
           "unknown property accepted");
  const pocl_kernel_wg_request invalid = { NULL, NULL, CL_KERNEL_LOCAL_MEM_SIZE,
                                         &output, device_maximum };
  require (pocl_kernel_wg_query (&invalid) == CL_INVALID_KERNEL,
           "null kernel accepted");
  require (value == 999, "invalid resolution wrote value");
  require (bytes == 123, "invalid resolution wrote size_ret");
}

static void test_device_maximum_delegation (void)
{
  Fixture f; initialize (&f);
  f.meta.max_workgroup_size = NULL;
  expect_scalar (&f, CL_KERNEL_WORK_GROUP_SIZE, 77);
  f.meta.max_workgroup_size = f.maximum; f.maximum[0] = 0;
  expect_scalar (&f, CL_KERNEL_WORK_GROUP_SIZE, 77);
  f.maximum_reply.status = CL_DEVICE_NOT_AVAILABLE;
  const pocl_kernel_wg_output output = { 0, NULL, NULL };
  require (query (&f, CL_KERNEL_WORK_GROUP_SIZE, &output) == CL_DEVICE_NOT_AVAILABLE,
           "device maximum error lost");
}

int main (void)
{
  test_scalar_properties ();
  test_local_dispatch_and_slots ();
  test_sizes_and_failed_writeback ();
  test_array_properties ();
  test_resolution_errors ();
  test_device_maximum_delegation ();
  puts ("kernel WG helpers: 6 host groups passed; injected replies, no public API");
  return 0;
}
