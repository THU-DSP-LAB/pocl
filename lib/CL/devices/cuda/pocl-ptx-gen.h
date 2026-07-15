/* pocl-ptx-gen.h - declarations for PTX code generator

   Copyright (c) 2016-2017 James Price / University of Bristol

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

#ifndef POCL_PTX_GEN_H
#define POCL_PTX_GEN_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Search for the libdevice bitcode library for the given GPU architecture. */
/* Returns zero on success, non-zero on failure. */
int findLibDevice(char LibDevicePath[PATH_MAX], const char *Arch);

/* Generate a PTX file from an LLVM bitcode file. */
/* Returns zero on success, non-zero on failure. */
typedef struct {
  void *device;
  void *program;
  void *llvm_module;
  const char *ptx_filename;
  unsigned ptx_version;
  const char *libdevice_path;
  void **alignment_map;
} pocl_ptx_gen_options_t;

int pocl_ptx_gen(const pocl_ptx_gen_options_t *options);

int pocl_cuda_create_alignments(void *llvm_module, void *program,
                                void **AlignmentMapPtr);

void pocl_cuda_destroy_alignments(void *llvm_module, void *AlignmentMapPtr);

typedef struct {
  void *llvm_module;
  const char *kernel_name;
  size_t *alignments;
  void *alignment_map;
} pocl_cuda_alignment_query_t;

/* Populate the alignment array with each kernel argument's requirements. */
int pocl_cuda_get_ptr_arg_alignment(const pocl_cuda_alignment_query_t *query);

int pocl_cuda_define_sub_group_size(void *llvm_module, void *program,
                                    int SGSize);

#ifdef __cplusplus
}
#endif

#endif /* POCL_PTX_GEN_H */
