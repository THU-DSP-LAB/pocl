/* OpenCL runtime library: clGetKernelWorkGroupInfo()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "devices/devices.h"
#include "pocl_util.h"

#include "pocl_kernel_wg_info.inc"

static cl_int
pocl_kernel_wg_device_maximum (cl_device_id device,
                              const pocl_kernel_wg_output *output)
{
  return POname (clGetDeviceInfo) (device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                  output->capacity, output->value,
                                  output->size_ret);
}

/* The external OpenCL API requires these six parameters. */
extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelWorkGroupInfo)
(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
 size_t param_value_size, void *param_value, size_t *param_value_size_ret)
  CL_API_SUFFIX__VERSION_1_0
{
  const pocl_kernel_wg_output output = {
    param_value_size, param_value, param_value_size_ret
  };
  const pocl_kernel_wg_request request = {
    kernel, device, param_name, &output, pocl_kernel_wg_device_maximum
  };
  return pocl_kernel_wg_query (&request);
}
POsym(clGetKernelWorkGroupInfo)
