/* ventus.c - a pocl device driver for ventus gpgpu

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2021 Pekka Jääskeläinen

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

#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"
#include "loadelf.hpp"

#include <any>
#include <assert.h>
#include <cstdint>
#include <ctype.h>
#include <limits.h>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utlist.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>

#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"
#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

  // from driver/include/ventus.h
//#if !defined(ENABLE_LLVM)
#include "ventus.h"
#include "pocl_ventus.h"
//#endif

#define VENTUS_INSTALL_PREFIX_DIR getenv("VENTUS_INSTALL_PREFIX")

  /* ENABLE_LLVM means to compile the kernel using pocl compiler,
 but for ventus(ventus has its own LLVM) it should be OFF. */


#define KNL_ENTRY 0
#define KNL_ARG_BASE 4
#define KNL_WORK_DIM 8
#define KNL_GL_SIZE_X 12
#define KNL_GL_SIZE_Y 16
#define KNL_GL_SIZE_Z 20
#define KNL_LC_SIZE_X 24
#define KNL_LC_SIZE_Y 28
#define KNL_LC_SIZE_Z 32
#define KNL_GL_OFFSET_X 36
#define KNL_GL_OFFSET_Y 40
#define KNL_GL_OFFSET_Z 44
#define KNL_PRINT_ADDR 48
#define KNL_PRINT_SIZE 52
#define KNL_MAX_METADATA_SIZE 64

// FIXME: Do not use hardcoded library search path!
static const char *ventus_final_ld_flags[] = {
  "-nodefaultlibs ",
  "-Wl,",
  VENTUS_INSTALL_PREFIX_DIR,
  "/lib/crt0.o ",
  "-Wl,",
  VENTUS_INSTALL_PREFIX_DIR,
  "/lib/riscv32clc.o ",
  "-Wl,--gc-sections ",
  "-L",
  VENTUS_INSTALL_PREFIX_DIR,
  "/lib ",
  "-lworkitem ",
  NULL
};

static const char *ventus_other_compile_flags[] = {
  "-I",
  VENTUS_INSTALL_PREFIX_DIR,
  "include/clc ",
  "-O1 ",
  "-Wl,-T,",
  VENTUS_INSTALL_PREFIX_DIR,
  "/lib/ldscripts/ventus/elf32lriscv.ld ",
  NULL
};

static const char *ventus_objdump_flags[] = {
  "-d",
  "--mattr=+v",
  NULL
};

static std::map<uint64_t, uint> program_ids;

void
pocl_ventus_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "ventus";
  ops->probe = pocl_ventus_probe;

  ops->uninit = pocl_ventus_uninit;
  ops->reinit = pocl_ventus_reinit;
  ops->init = pocl_ventus_init;

  ops->alloc_mem_obj = pocl_ventus_alloc_mem_obj;
  ops->free = pocl_ventus_free;

  ops->read = pocl_ventus_read;
  ops->read_rect = pocl_ventus_read_rect;
  ops->write = pocl_ventus_write;
  ops->write_rect = pocl_ventus_write_rect;

  ops->run = pocl_ventus_run;
  ops->run_native = NULL;
  /***********************No need to modify for now*************************/

  ops->copy = pocl_ventus_driver_copy;
  ops->copy_with_size = NULL;
  ops->copy_rect = pocl_ventus_copy_rect;

  ops->memfill = pocl_ventus_memfill;
  ops->map_mem = pocl_ventus_map_mem;
  ops->unmap_mem = pocl_ventus_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = NULL;

  /* for ventus,pocl does not need to compile the kernel,so they are set to NULL */
  ops->build_source = pocl_ventus_build_source;
  ops->post_build_program = pocl_ventus_post_build_program;
  ops->link_program = NULL;
  ops->build_binary = NULL;
  ops->free_program = NULL;
  ops->setup_metadata = pocl_ventus_setup_metadata;
  ops->supports_binary = NULL;
  ops->build_poclbinary = NULL;
  ops->compile_kernel = NULL;  //or use int (*build_builtin) (cl_program program, cl_uint device_i);

  ops->join = pocl_ventus_join;
  ops->submit = pocl_ventus_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_ventus_notify;
  ops->flush = pocl_ventus_flush;

  ops->build_hash = pocl_ventus_build_hash;
  ops->compute_local_size = NULL;

  ops->get_device_info_ext = NULL;

  // Currently ventus does not support svm
  ops->svm_free = NULL;
  ops->svm_alloc = NULL;
  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = NULL;
  ops->svm_fill = NULL;

  ops->create_kernel = NULL;
  ops->free_kernel = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;

  ops->can_migrate_d2d = NULL;
  ops->migrate_d2d = NULL;

  ops->copy_image_rect = NULL;
  ops->write_image_rect = NULL;
  ops->read_image_rect = NULL;
  ops->map_image = NULL;
  ops->unmap_image = NULL;
  ops->fill_image = NULL;
}

char *
pocl_ventus_build_hash (cl_device_id device)
{
  char* res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "THU-ventus-GPGPU\n");
  return res;
}

unsigned int
pocl_ventus_probe(struct pocl_device_ops *ops)
{
  if (0 == strcmp(ops->device_name, "ventus"))
    return 1;
  return 0;
}

uint64_t get_env_u64(const char* varname, uint64_t default_val) {
    char* numVar = std::getenv(varname);
    if (numVar) {
      uint64_t val;
      try {
        val = std::stoul(numVar);
      } catch (...) {
        POCL_MSG_ERR("environment variable %s is not a valid integer, use default value %lu\n", varname, default_val);
        return default_val;
      }
      return val;
    } else {
        POCL_MSG_ERR("environment variable %s is not found, use default value %lu\n", varname, default_val);
        return default_val;
    }
};

cl_int
pocl_ventus_init (unsigned j, cl_device_id dev, const char* parameters)
{
  struct vt_device_data_t *d;
  cl_int ret = CL_SUCCESS;
  int err;

  d = (struct vt_device_data_t *) calloc (1, sizeof (struct vt_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  vt_device_h vt_device;

  err = vt_dev_open(&vt_device);
  if (err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }

  /*
  // add storage for position pointer
  uint32_t print_buf_dev_size = PRINT_BUFFER_SIZE + sizeof(uint32_t);
  size_t vt_print_buf_d;
  err = vt_buf_alloc(vt_device, print_buf_dev_size, &vt_print_buf_d);
  if (err != 0) {
    vt_dev_close(vt_device);
    free(d);
    return CL_INVALID_DEVICE;
  }  */

  d->vt_device   = vt_device;

  d->current_kernel = NULL;

  dev->data = d;

  SETUP_DEVICE_CL_VERSION(2, 0);
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->long_name = "Ventus GPGPU device";
  dev->vendor = "THU";
  dev->vendor_id = 0; // TODO: Update vendor id!
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;

  char extensions[1024];
  extensions[0] = 0;
  strcat (extensions, "cl_khr_fp64");
  dev->extensions = strdup (extensions);


  char features[1024];
  features[0] = 0;
  strcat (features, " __opencl_c_generic_address_space"
                    " __opencl_c_named_address_space_builtins");
  dev->features = strdup (features);

  dev->profile = "EMBEDDED_PROFILE";
  dev->endian_little = CL_TRUE;

  dev->max_mem_alloc_size = 500 * 1024 * 1024; //100M
  dev->mem_base_addr_align = 128;

  dev->max_constant_buffer_size = 32768;     // TODO: Update this to conformant to OCL 2.0 // no cpmstant buffer now
  dev->local_mem_size = 64 * 1024;     // TODO: Update this to conformant to OCL 2.0 // 64kB per SM
  dev->global_mem_size = 1024 * 1024 * 1024; // 1G ram
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;
  dev->global_mem_cacheline_size = 128; // 128 Bytes for 32 thread
  dev->global_mem_cache_size = 64 * 128;
  dev->image_max_buffer_size = dev->max_mem_alloc_size / 16;

  dev->image2d_max_width = 1024; // TODO: Update
  dev->image3d_max_width = 1024; // TODO: Update

  dev->max_work_item_dimensions = 3;
  // RTL and cyclesim: repo default 8-warp 32-thread
  // try to load hardware info from hardware (simulator) first
  uint64_t num_warp = 8, num_thread = 32;
  if (vt_dev_caps(nullptr, VT_CAPS_MAX_WARPS, &num_warp) != 0) {
    num_warp = 8;
  }
  if (vt_dev_caps(nullptr, VT_CAPS_MAX_THREADS, &num_thread) != 0) {
    num_thread = 32;
  }
  // if env var is set, override the detected hardware config (mainly for spike)
  num_warp = get_env_u64("NUM_WARP", num_warp);
  num_thread = get_env_u64("NUM_THREAD", num_thread);
  dev->max_work_group_size = num_warp*num_thread;
  dev->max_work_item_sizes[0] = num_warp*num_thread;
  dev->max_work_item_sizes[1] = num_warp*num_thread;
  dev->max_work_item_sizes[2] = 64; // 对齐 NV CUDA 限制
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;
  dev->max_parameter_size = 1024;
  dev->max_constant_args = 8;
  dev->max_compute_units = 16 * 32; // 16 SM comprise 16 int32 and 16 fp32 cores
  dev->max_clock_frequency = 100; // TODO: This is frequency in MHz
  dev->address_bits = 32;

  // Supports device side printf
  dev->device_side_printf = 0;
  dev->printf_buffer_size = 0;

  dev->device_alloca_locals = 1; //

  // Doesn't support partition
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  // Doesn't support SVM
  dev->svm_allocation_priority = 0;

  dev->final_linkage_flags = ventus_final_ld_flags;

  // float rounding mode
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

  dev->double_fp_config = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST
                          | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF
                          | CL_FP_INF_NAN | CL_FP_DENORM;

  // TODO: Do we have builtin kernels for Ventus?

#ifdef ENABLE_LLVM
  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
#else
  dev->llvm_target_triplet = "";
  dev->llvm_cpu = "";
#endif


#if (!defined(ENABLE_CONFORMANCE))
  /* full memory consistency model for atomic memory and fence operations
  https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#opencl-3.0-backwards-compatibility */
  dev->atomic_memory_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE
                                       | CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
#endif

  POCL_INIT_LOCK (d->cq_lock);


  return ret;
}

#define PRINT_CHISEL_TESTCODE
// 用于生成metadata/data文件
std::vector<MemBlock> g_vt_dump_mem;
#ifdef PRINT_CHISEL_TESTCODE

void fp_write_file(FILE *fp, const void *p, uint64_t size){
  for (size_t i = 0; i < (size+sizeof(uint32_t)-1) / sizeof(uint32_t); ++i)
    fprintf(fp,"%08x\n",*((uint32_t*)p+i));
}
#endif

void
pocl_ventus_run (void *data, _cl_command_node *cmd)
{

  struct vt_device_data_t *d;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  int err;
  //calculating number of kernel name appears.
  static std::map<std::string, int> knl_name_list;
  auto it = knl_name_list.find(meta->name);
  if(it != knl_name_list.end())
      it->second++;
  else
      knl_name_list[meta->name] = 0;

  if(program_ids.find(uint64_t(kernel->program)) == program_ids.end()) {
      POCL_MSG_ERR("ERROR: program id not found\n");
      exit(10);
  }
  uint id = program_ids[uint64_t(kernel->program)];

    uint64_t num_thread = 32;
    if (vt_dev_caps(nullptr, VT_CAPS_MAX_THREADS, &num_thread) != 0) {
      num_thread = 32;
    }
    num_thread = get_env_u64("NUM_THREAD", num_thread);
    uint64_t num_warp=(pc->local_size[0]*pc->local_size[1]*pc->local_size[2] + num_thread-1)/ num_thread;
    uint64_t num_workgroups[3];
    num_workgroups[0]=pc->num_groups[0];num_workgroups[1]=pc->num_groups[1];num_workgroups[2]=pc->num_groups[2];
    uint64_t num_workgroup=num_workgroups[0]*num_workgroups[1]*num_workgroups[2];
    uint64_t num_processor=num_warp*num_workgroup;
    uint64_t ldssize=0x1000;
    uint64_t pdssize=0x1000;
    uint64_t pdsbase=0x8a000000;
    uint64_t start_pc=0x80000000;
    uint64_t knlbase=0x90000000;
    uint64_t sgpr_usage=64;
    uint64_t vgpr_usage=64;
    uint64_t new_lds_base = ventus_local_base+ventus_local_size_total;
    uint64_t local_arg[meta->num_args];

#ifdef PRINT_CHISEL_TESTCODE
    std::string metadata_name_s = std::string(meta->name)+"_"+std::to_string(knl_name_list[meta->name])+".metadata";
    const char *c_metadata_name = metadata_name_s.c_str();
    std::string data_name_s = std::string(meta->name)+"_"+std::to_string(knl_name_list[meta->name])+".data";
    const char *c_data_name = data_name_s.c_str();
    FILE *fp_metadata=fopen(c_metadata_name,"w");
    FILE *fp_data=fopen(c_data_name,"w");
#endif

/*
step1 upload kernel_rom & allocate its mem (load cache file?)
step2 allocate kernel argument & arg buffer
      notice kernel arg buffer is offered by command.run.
step3 prepare kernel metadata (pc(start_pc=8000) & kernel entrance(0x8000005c) & arg pointer )
step4 prepare driver metadata
step5 make a writefile for chisel
*/
  assert(cmd->device->data != NULL);
  d = (struct vt_device_data_t *)cmd->device->data;


  // preprocess some kernel arguments
  std::vector<std::any> args(meta->num_args + meta->num_locals);
  //TODO 1: support local buffer as argument. Notice in current structure, allocated localmembuffer will be mapped to ddr space.
  //TODO 2: print buffer support in pocl
  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i)
    {
      pocl_argument* al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i]))
        {
          if (cmd->device->device_alloca_locals)
            {
                 /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
                // TODO: __local__ arg not supported yet,
                // but can be used on spike (treated as global memory)
                void* tmp_arg = malloc(al->size);
                memset(tmp_arg,0,al->size);
                if(al->value != nullptr)
                    memcpy(tmp_arg, al->value, al->size);
                uint64_t aligned_size = (al->size / 4096 + 1)*4096;
                new_lds_base -= aligned_size;
                memcpy(&local_arg[i], &new_lds_base, sizeof(uint32_t));
                POCL_MSG_PRINT_VENTUS("new_lds_base:%08lx\n", new_lds_base);
                err = vt_copy_to_dev(d->vt_device, new_lds_base, tmp_arg, aligned_size,0,0);
                if (err != 0) {
                    abort();
                }
                #ifdef PRINT_CHISEL_TESTCODE
                    // assert(0); // Not support local buffer arg yet.
                    g_vt_dump_mem.emplace_back(new_lds_base, aligned_size);
                    g_vt_dump_mem.back().data.assign(al->size, 0);
                #endif
              POCL_MSG_WARN("not support local buffer arg yet.\n");
              args[i] = uint64_t(new_lds_base); // device-side ptr
            }
          else
            {
              // what does this do?
              assert(0);
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL) {
              args[i] = uint64_t(0); // device-side nullptr
          } else {
              if (al->is_svm) {
                  args[i] = al->value; // host-side SVM pointer
              } else { // device-side pointer
                  cl_mem m = (*(cl_mem *)(al->value));
                  args[i] = reinterpret_cast<uint64_t>(m->device_ptrs[cmd->device->global_mem_id].mem_ptr) + al->offset;
              }
          }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          pocl_fill_dev_image_t (&di, al, cmd->device);
          args[i] = di;
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          pocl_fill_dev_sampler_t (&ds, al);
          args[i] = ds;
        }
      else
        {
          // other type: do noting, just use al->value later
        }
    }

  if (cmd->device->device_alloca_locals)
    {
      POCL_MSG_WARN("notice that ventus hasn't support local buffer as argument yet.\n");
      /* Local buffers are allocated in the device side work-group
         launcher. Let's pass only the sizes of the local args in
         the arg buffer. */
      for (i = 0; i < meta->num_locals; ++i)
        {
          size_t s = meta->local_sizes[i]; //TODO: create local_buf at ddr, and map argument to this addr.
          size_t j = meta->num_args + i;
        }
    }
  else
    {
      for (i = 0; i < meta->num_locals; ++i)
        {
          size_t s = meta->local_sizes[i];
          size_t j = meta->num_args + i;
          // TODO: check this. should alloc on device-side?
          // arguments[j] = malloc (sizeof (void *));
          // void *pp = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, s);
          // *(void **)(arguments[j]) = pp;
        }
    }

  /*pc->printf_buffer = d->printf_buffer;
  assert (pc->printf_buffer != NULL);
  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert (pc->printf_buffer_capacity > 0);
  uint32_t position = 0;
  pc->printf_buffer_position = &position;
  pc->global_var_buffer = program->gvar_storage[dev_i];*/

  /**********************************************************************************************************
   * create argument buffer now.
   * The variable `arg_dev_mem_addr` reveals the address that kernel argument
   * buffer saves in device memory, to get the content of kernel argument,
   * check the variable `abuf_args_data`.
   **********************************************************************************************************/
  constexpr uint8_t device_ptr_size = 4;
  uint64_t abuf_size = 0;
  for (i = 0; i < meta->num_args; ++i) {
      pocl_argument* al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i]) && cmd->device->device_alloca_locals) {
        abuf_size += device_ptr_size;
      } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        abuf_size += al->is_svm ? sizeof(void*) : device_ptr_size;
      } else if((meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        abuf_size += device_ptr_size; // TODO
      } else {
        abuf_size = pocl_align_value(abuf_size+al->size, pocl_size_ceil2_64(std::max(al->size, (uint64_t)4)));
      }
  }

  assert(abuf_size <= 0xffff);
  char* abuf_args_data = (char*)malloc(abuf_size);
  uint64_t abuf_args_p = 0;
  for(i = 0; i < meta->num_args; ++i) {
      pocl_argument* al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i]) && cmd->device->device_alloca_locals) {
        auto devptr = std::any_cast<uint64_t>(args[i]);
        abuf_args_p = pocl_align_value(abuf_args_p, device_ptr_size);
        memcpy(abuf_args_data + abuf_args_p, &devptr, device_ptr_size);
        abuf_args_p += device_ptr_size;
      } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER){
        uint64_t devptr;
        uintptr_t svmptr;
        if (al->is_svm) {
            svmptr = (uintptr_t) std::any_cast<void*>(args[i]);
            assert(0); // not support SVM now
        } else {
            devptr = std::any_cast<uint64_t>(args[i]);
        }
        abuf_args_p = pocl_align_value(abuf_args_p, device_ptr_size);
        memcpy(abuf_args_data + abuf_args_p, &devptr, device_ptr_size);
        abuf_args_p += device_ptr_size;
      } else if((meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        assert(0); // not support
        // maybe: vt_buf_alloc + vt_copy_to_dev image/sampler struct and pass device ptr as args
        abuf_args_p = pocl_align_value(abuf_args_p, device_ptr_size);
        // memcpy(abuf_args_data+alloc_p,arguments[i],device_ptr_size);
        abuf_args_p += device_ptr_size;
      } else {
        abuf_args_p = pocl_align_value(abuf_args_p, pocl_size_ceil2_64(std::max(al->size, (uint64_t)4)));
        memcpy(abuf_args_data+abuf_args_p,al->value,al->size);
        abuf_args_p += al->size;
      }
  }
  POCL_MSG_PRINT_VENTUS("Allocating kernel arg buffer entry:\n");
  uint64_t arg_dev_mem_addr;
  if (abuf_size == 0) {
    arg_dev_mem_addr = 0;
  } else {
    err = vt_buf_alloc(d->vt_device, abuf_size, &arg_dev_mem_addr,0,0,0);
    if (err != 0) {
      POCL_MSG_ERR("ERROR: vt_buf_alloc failed\n");
      abort();
    }
  }

  #ifdef PRINT_CHISEL_TESTCODE
    g_vt_dump_mem.emplace_back(arg_dev_mem_addr, abuf_size);
    g_vt_dump_mem.back().data.assign(abuf_args_data, abuf_args_data + abuf_size);
  #endif

  if (abuf_size > 0) {
    err = vt_copy_to_dev(d->vt_device,arg_dev_mem_addr,abuf_args_data, abuf_size, 0,0);
    if (err != 0) {
      POCL_MSG_ERR("ERROR: vt_copy_to_dev failed\n");
      abort();
    }
  }
  /**********************************************************************************************************/

  //after checking pocl_cache_binary, use the following to pass in.
   /*if (NULL == d->current_kernel || d->current_kernel != kernel) {
       d->current_kernel = kernel;
      char program_bin_path[POCL_FILENAME_LENGTH];
      pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
      err = vt_upload_kernel_file(d->vt_device, program_bin_path,0);
      assert(0 == err);
    }*/

  /**********************************************************************************************************
   * Resolve kernel function entry address from the ELF file.
   * clCreateKernel.c line 79
   ***********************************************************************************************************/
	uint32_t kernel_entry;
  char filename[256] = "object";
  strcat(filename, std::to_string(id).c_str());
  char binary_filename[256];
  strcpy(binary_filename, filename);
  strcat(binary_filename, ".riscv");
#ifdef __linux__
  {
    auto sym = get_symbol_value_from_elf(binary_filename, meta->name, nullptr);
    if (!sym.has_value()) {
      POCL_MSG_ERR("ELF: cannot find kernel symbol '%s' from '%s'\n", meta->name,
                   binary_filename);
      abort();
    }
    if (*sym > UINT32_MAX) {
      POCL_MSG_ERR("ELF: kernel symbol '%s' address 0x%lx out of 32-bit range\n",
                   meta->name, (unsigned long)*sym);
      abort();
    }
    kernel_entry = (uint32_t)(*sym);
    POCL_MSG_PRINT_VENTUS("Kernel entry of \"%s\" is : \"0x%x\"\n", kernel->name,
                          kernel_entry);
  }
#else
  POCL_MSG_ERR("This operating system is not supported now by ventus, please use linux!\n");
  abort();
#endif
  /***********************************************************************************************************/



  uint64_t pc_src_size=0x10000000;
  uint64_t pc_dev_mem_addr = 0x80000000;

  // Upload the kernel ELF to Ventus driver
  vt_upload_kernel_file(d->vt_device, binary_filename, 0);
  #ifdef PRINT_CHISEL_TESTCODE
	fp_write_file(fp_metadata, &(pc_dev_mem_addr), sizeof(uint64_t));
  std::vector<MemBlock> elf_data = get_data_from_elf(binary_filename, nullptr);
  g_vt_dump_mem.insert(g_vt_dump_mem.end(), elf_data.begin(), elf_data.end());
  #endif
  /***********************************************************************************************************/


//prepare privatemem
  uint64_t pds_src_size=pdssize*num_thread*num_warp*num_workgroup;

  // Now spike run workgroups one by one, to fix the over 4G issue
  // Spike specified the pdssize to 0x10000000 for each workgroup now
  if (pds_src_size > 0x10000000) {
    pds_src_size = 0x10000000;
    POCL_MSG_PRINT_VENTUS("pdssize setting to 0x%x\n", 0x10000000);
  }

  uint64_t pds_dev_mem_addr;
    POCL_MSG_PRINT_VENTUS("Preparing private memory of ventus:\n");
  err = vt_buf_alloc(d->vt_device, pds_src_size, &pds_dev_mem_addr,0,0,0);
  if (err != 0) {
    abort();
  }
  #ifdef PRINT_CHISEL_TESTCODE
    g_vt_dump_mem.emplace_back(pds_dev_mem_addr, pds_src_size);
  #endif



//prepare kernel_metadata
  char *kernel_metadata= (char*)malloc(sizeof(char)*KNL_MAX_METADATA_SIZE);
  memset(kernel_metadata,0,KNL_MAX_METADATA_SIZE);
  memcpy(kernel_metadata+KNL_ENTRY,&kernel_entry,4);
  uint32_t arg_dev_mem_addr_32=(uint32_t)arg_dev_mem_addr;
  memcpy(kernel_metadata+KNL_ARG_BASE,&arg_dev_mem_addr_32,4);
  memcpy(kernel_metadata+KNL_WORK_DIM,&(pc->work_dim),4);
  uint32_t local_size_32[3];local_size_32[0]=(uint32_t)pc->local_size[0];local_size_32[1]=(uint32_t)pc->local_size[1];local_size_32[2]=(uint32_t)pc->local_size[2];
  uint32_t global_offset_32[3];global_offset_32[0]=(uint32_t)pc->global_offset[0];global_offset_32[1]=(uint32_t)pc->global_offset[1];global_offset_32[2]=(uint32_t)pc->global_offset[2];
  uint32_t global_size_32[3];global_size_32[0]=(uint32_t)pc->num_groups[0]*local_size_32[0];global_size_32[1]=(uint32_t)pc->num_groups[1]*local_size_32[1];global_size_32[2]=(uint32_t)pc->num_groups[2]*local_size_32[2];
  memcpy(kernel_metadata+KNL_GL_SIZE_X,&global_size_32[0],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Y,&global_size_32[1],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Z,&global_size_32[2],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_X,&local_size_32[0],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Y,&local_size_32[1],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Z,&local_size_32[2],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_X,&global_offset_32[0],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Y,&global_offset_32[1],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Z,&global_offset_32[2],4);
//memcpy(kernel_metadata+KNL_PRINT_ADDR,global_offset_32[0],4);
    POCL_MSG_PRINT_VENTUS("Allocating metadata space:\n");
  uint64_t knl_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, KNL_MAX_METADATA_SIZE, &knl_dev_mem_addr,0,0,0);
  if (err != 0) {
    abort();
  }
  err = vt_copy_to_dev(d->vt_device,knl_dev_mem_addr,kernel_metadata, KNL_MAX_METADATA_SIZE, 0,0);
  if (err != 0) {
    abort();
  }
  POCL_MSG_PRINT_VENTUS("kernel metadata has been written to 0x%lx\n", knl_dev_mem_addr);
  #ifdef PRINT_CHISEL_TESTCODE
  g_vt_dump_mem.emplace_back(knl_dev_mem_addr, KNL_MAX_METADATA_SIZE, std::vector<uint8_t>(kernel_metadata, kernel_metadata + KNL_MAX_METADATA_SIZE));
  #endif


  pdsbase=pds_dev_mem_addr;
  knlbase=knl_dev_mem_addr;
  vt_kernel_metadata_t driver_meta;
    driver_meta.kernel_id=0;
    driver_meta.kernel_size[0]=num_workgroups[0];
    driver_meta.kernel_size[1]=num_workgroups[1];
    driver_meta.kernel_size[2]=num_workgroups[2];
    driver_meta.wf_size=num_thread;
    driver_meta.wg_size=num_warp;
    driver_meta.metaDataBaseAddr=knlbase;
    driver_meta.ldsSize=ldssize;
    driver_meta.pdsSize=pdssize;
    driver_meta.sgprUsage=sgpr_usage;
    driver_meta.vgprUsage=vgpr_usage;
    driver_meta.pdsBaseAddr=pdsbase;
    for (i = 0; i < 3; ++i) {
      driver_meta.num_thread_global[i] = global_size_32[i];
      driver_meta.num_thread_local[i] = local_size_32[i];
      driver_meta.threadIdxOffset[i] = global_offset_32[i];
    }
    driver_meta.kernel_name=meta->name;

// prepare a write function

  #ifdef PRINT_CHISEL_TESTCODE
    fp_write_file(fp_metadata,&(driver_meta.kernel_id),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[0]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[1]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[2]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.wf_size),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.wg_size),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.metaDataBaseAddr),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.ldsSize),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.pdsSize),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.sgprUsage),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.vgprUsage),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.pdsBaseAddr),sizeof(uint64_t));
    uint64_t num_buffer = g_vt_dump_mem.size();
    fp_write_file(fp_metadata, &num_buffer, sizeof(uint64_t));
    for (const auto& buf : g_vt_dump_mem) {
      fp_write_file(fp_metadata, &buf.vaddr, sizeof(uint64_t));
    }
    for (const auto& buf : g_vt_dump_mem) {
      uint64_t data_size = buf.data.size();
      fp_write_file(fp_metadata, &data_size, sizeof(uint64_t));
      fp_write_file(fp_data, buf.data.data(), buf.data.size());
    }
    for (const auto& buf : g_vt_dump_mem) {
      fp_write_file(fp_metadata, &buf.memsz, sizeof(uint64_t));
    }
    g_vt_dump_mem.clear();
    fclose(fp_metadata);
    fclose(fp_data);
  #endif



//pass metadata to "run"
  // quick off kernel execution
  err = vt_start(d->vt_device, &driver_meta,0);
  assert(0 == err);

  // wait for the execution to complete
  err = vt_ready_wait(d->vt_device, 1000);
  assert(0 == err);

  // move print buffer back or wait to read?

    // rename log file from spike and add index for log
    char sp_logname[256];
    strcpy(sp_logname, filename);
    strcat(sp_logname, ".riscv.log");
    char newName[256]; // 假设文件名不超过 255 个字符
    FILE* logfp = fopen(sp_logname, "r");
    if(logfp) {
        fclose(logfp);
        strcpy(newName, meta->name);
        sprintf(newName, "%s_%d.log",meta->name,knl_name_list[meta->name]);
        //strcat(newName, ".log");
        if(rename(sp_logname, newName) == 0) {
            POCL_MSG_PRINT_VENTUS("Log file %s renamed successfully to %s.\n", sp_logname, newName);
        } else {
            POCL_MSG_PRINT_VENTUS("Unable to rename the log file %s.\n", sp_logname);
        }
    }

    err |= vt_one_buf_free(d->vt_device, KNL_MAX_METADATA_SIZE, &knl_dev_mem_addr, 0, 0);
    err |= vt_one_buf_free(d->vt_device, pds_src_size, &pds_dev_mem_addr, 0, 0);
    if (abuf_size > 0) {
      err |= vt_one_buf_free(d->vt_device, abuf_size, &arg_dev_mem_addr, 0, 0);
    }
    assert(0 == err);
  free(abuf_args_data);
  free(kernel_metadata);

  //pocl_release_dlhandle_cache(cmd);

}


cl_int
pocl_ventus_uninit (unsigned j, cl_device_id device)
{
  struct vt_device_data_t *d = (struct vt_device_data_t*)device->data;
  if (NULL == d)
    return CL_SUCCESS;

  vt_dev_close(d->vt_device);

  POCL_DESTROY_LOCK(d->cq_lock);
  POCL_MEM_FREE(d);
  device->data = NULL;

  return CL_SUCCESS;
}

cl_int
pocl_ventus_reinit (unsigned j, cl_device_id device)
{
  struct vt_device_data_t *d;
  int err;

  d = (struct vt_device_data_t *) calloc (1, sizeof (struct vt_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  vt_device_h vt_device;
  err = vt_dev_open(&vt_device);
  if (err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }

  d->vt_device = vt_device;
  d->current_kernel = NULL;

  POCL_INIT_LOCK (d->cq_lock);
  device->data = d;

  return CL_SUCCESS;
}

void ventus_command_scheduler (struct vt_device_data_t *d)
{
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (d->cq_lock);
    }

  return;
}

void
pocl_ventus_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)node->device->data;

  //if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
  //  pocl_check_kernel_dlhandle_cache (node, 1, 1);

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_ventus_flush (cl_device_id device, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_ventus_join (cl_device_id device, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_ventus_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          POCL_LOCK (d->cq_lock);
          CDL_DELETE (d->command_list, node);
          CDL_PREPEND (d->ready_list, node);
          ventus_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

void
pocl_ventus_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize)
{
  POCL_MSG_PRINT_VENTUS("in pocl ventus compile kernel func\n");
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (cmd, 0, specialize);
}

void pocl_ventus_free(cl_device_id device, cl_mem memobj) {
  cl_mem_flags flags = memobj->flags;
  vt_device_data_t* d = (vt_device_data_t *)device->data;
  uint64_t dev_mem_addr = reinterpret_cast<uint64_t>(memobj->device_ptrs[device->dev_id].mem_ptr);

  /* The host program can provide the runtime with a pointer
  to a block of continuous memory to hold the memory object
  when the object is created (CL_MEM_USE_HOST_PTR).
  Alternatively, the physical memory can be managed
  by the OpenCL runtime and not be directly accessible
  to the host program.*/
  if (flags & CL_MEM_USE_HOST_PTR) {
    vt_one_buf_free(d->vt_device,memobj->size,&dev_mem_addr,0,0);
  } else {
    vt_one_buf_free(d->vt_device,memobj->size,&dev_mem_addr,0,0);
    free(memobj->mem_host_ptr);
    memobj->mem_host_ptr = NULL;
  }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
  memobj->device_ptrs[device->dev_id].mem_ptr = nullptr;
}


cl_int
pocl_ventus_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {

  cl_mem_flags flags = mem_obj->flags;
  vt_device_data_t* d = (vt_device_data_t *)device->data;
  pocl_global_mem_t *mem = device->global_memory;
  int err;
  uint64_t dev_mem_addr;
  // if this memory object has not been allocated device memory space,
  // then allocating a device memory and binding the memory pointer to cl_mem object
  assert(sizeof(mem_obj->device_ptrs[device->dev_id].mem_ptr) >= sizeof(uint64_t));
  err = vt_buf_alloc(d->vt_device, mem_obj->size, &dev_mem_addr,0,0,0);
  if (err != 0 || dev_mem_addr == 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }
  mem_obj->device_ptrs[device->dev_id].mem_ptr = reinterpret_cast<void*>(dev_mem_addr);
  assert(!std::any_of(g_vt_dump_mem.begin(), g_vt_dump_mem.end(),
             [dev_mem_addr](const MemBlock &mb) {
               return mb.vaddr == dev_mem_addr;
             }));
  g_vt_dump_mem.emplace_back(dev_mem_addr, mem_obj->size);

  // if the memory object has been allocated device memory pointer and
  // if the flags indicates that copy data from host ptr, then do the following operations.
  if ((flags & CL_MEM_COPY_HOST_PTR) && mem_obj->device_ptrs[device->dev_id].mem_ptr) {
    if (mem_obj->mem_host_ptr) {
      err = vt_copy_to_dev(d->vt_device,
                           reinterpret_cast<uint64_t>(mem_obj->device_ptrs[device->dev_id].mem_ptr),
                           mem_obj->mem_host_ptr, mem_obj->size, 0, 0);
      if (err != 0) {
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
      g_vt_dump_mem.back().data.assign((uint8_t*)mem_obj->mem_host_ptr, (uint8_t*)mem_obj->mem_host_ptr + mem_obj->size);
    }
  }

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
//    abort(); // TODO
  }

  return CL_SUCCESS;
}

void pocl_ventus_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset,
                      size_t size) {
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;
  int err = vt_copy_from_dev(
      d->vt_device, reinterpret_cast<uint64_t>(src_mem_id->mem_ptr) + offset,
      host_ptr, size, 0, 0);
  assert(0 == err);
}

void pocl_ventus_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset,
                       size_t size) {
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;
  int err = vt_copy_to_dev(
      d->vt_device, reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr) + offset,
      host_ptr, size, 0, 0);
  assert(0 == err);
#ifdef PRINT_CHISEL_TESTCODE
  uint64_t dev_addr = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr) + offset;
  auto memblk = std::find_if(g_vt_dump_mem.begin(), g_vt_dump_mem.end(),
                     [dev_addr, size](const MemBlock &mb) {
                       return mb.vaddr == dev_addr && mb.memsz >= size;
                     });
  if (memblk != g_vt_dump_mem.end()) {
    memblk->data.assign((uint8_t*)host_ptr, (uint8_t*)host_ptr + size);
  } else {
    g_vt_dump_mem.emplace_back(dev_addr, size);
    g_vt_dump_mem.back().data.assign((uint8_t*)host_ptr, (uint8_t*)host_ptr + size);
  }
#endif // PRINT_CHISEL_TESTCODE
}

void
pocl_ventus_driver_copy (void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                  pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                  size_t dst_offset, size_t src_offset, size_t size)
{
    vt_device_data_t *d = (vt_device_data_t *)data;
    char *__restrict__ src_ptr = (char *)src_mem_id->mem_ptr;
    char *__restrict__ dst_ptr = (char *)dst_mem_id->mem_ptr;

    void *host_ptr = malloc(size);
    uint64_t dev_src_addr = reinterpret_cast<uint64_t>(src_mem_id->mem_ptr);
    uint64_t dev_dst_addr = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr);

    int err = vt_copy_from_dev(d->vt_device, dev_src_addr + src_offset, host_ptr, size, 0, 0);
    assert(0 == err);

    err = vt_copy_to_dev(d->vt_device, dev_dst_addr + dst_offset, host_ptr, size, 0, 0);
    assert(0 == err);

    free(host_ptr);
}

void
pocl_ventus_read_rect (void *data, void *__restrict__ const host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin,
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
    vt_device_data_t *d = (vt_device_data_t *)data;
    uint64_t src_addr = reinterpret_cast<uint64_t>(src_mem_id->mem_ptr);

    char *buff = (char *)malloc(src_buf->size);
    int err = vt_copy_from_dev(d->vt_device, src_addr, buff, src_buf->size, 0, 0);
    assert(0 == err);

    char const *__restrict const adjusted_device_ptr
        = buff + buffer_origin[2] * buffer_slice_pitch
        + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
    char *__restrict__ const adjusted_host_ptr
        = (char *)host_ptr + host_origin[2] * host_slice_pitch
        + host_origin[1] * host_row_pitch + host_origin[0];
    size_t j, k;

    /* TODO: handle overlaping regions */
    if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
                && (buffer_slice_pitch == host_slice_pitch
                && host_slice_pitch == (region[1] * region[0]))) {
        memcpy (adjusted_host_ptr, adjusted_device_ptr,
                region[2] * region[1] * region[0]);
    } else {
        for (k = 0; k < region[2]; ++k)
            for (j = 0; j < region[1]; ++j)
                memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
                        adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
                        region[0]);
    }
    free(buff);
}

void
pocl_ventus_write_rect (void *data, const void *__restrict__ const host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        const size_t *__restrict__ const buffer_origin,
                        const size_t *__restrict__ const host_origin,
                        const size_t *__restrict__ const region,
                        size_t const buffer_row_pitch,
                        size_t const buffer_slice_pitch,
                        size_t const host_row_pitch,
                        size_t const host_slice_pitch)
{
    vt_device_data_t *d = (vt_device_data_t *)data;
    uint64_t dev_addr = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr);

    char *buff = (char *)malloc(dst_buf->size);
    int err = vt_copy_from_dev(d->vt_device, dev_addr, buff, dst_buf->size, 0, 0);
    assert(0 == err);

    char *__restrict const adjusted_device_ptr
        = buff + buffer_origin[0]
        + buffer_row_pitch * buffer_origin[1]
        + buffer_slice_pitch * buffer_origin[2];
    char const *__restrict__ const adjusted_host_ptr
        = (char const *)host_ptr + host_origin[0]
        + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];
    size_t j, k;

    /* TODO: handle overlaping regions */
    if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
                && (buffer_slice_pitch == host_slice_pitch
                && host_slice_pitch == (region[1] * region[0]))) {
        memcpy (adjusted_device_ptr, adjusted_host_ptr, region[2] * region[1] * region[0]);
    } else {
        for (k = 0; k < region[2]; ++k) {
            for (j = 0; j < region[1]; ++j) {
                memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
                        adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
                        region[0]);
            }
        }
    }

    err = vt_copy_to_dev(d->vt_device, dev_addr, buff, dst_buf->size, 0, 0);
    assert(0 == err);
    free(buff);
}

void
pocl_ventus_copy_rect (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                       cl_mem src_buf,
                       const size_t *__restrict__ const dst_origin,
                       const size_t *__restrict__ const src_origin,
                       const size_t *__restrict__ const region,
                       size_t const dst_row_pitch,
                       size_t const dst_slice_pitch,
                       size_t const src_row_pitch,
                       size_t const src_slice_pitch)
{
    int err;
    vt_device_data_t *d = (vt_device_data_t *)data;
    uint64_t dev_src_addr  = reinterpret_cast<uint64_t>(src_mem_id->mem_ptr);
    uint64_t dev_dst_addr  = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr);
    char *src_buff = (char *)malloc(src_buf->size);
    char *dst_buff = (char *)malloc(dst_buf->size);

    err = vt_copy_from_dev(d->vt_device, dev_src_addr, src_buff, src_buf->size, 0, 0);
    assert(0 == err);
    err = vt_copy_from_dev(d->vt_device, dev_dst_addr, dst_buff, dst_buf->size, 0, 0);
    assert(0 == err);

    char const *__restrict const adjusted_src_ptr
        = (char const *)src_buff + src_origin[0] + src_row_pitch * src_origin[1]
        + src_slice_pitch * src_origin[2];
    char *__restrict__ const adjusted_dst_ptr
        = (char *)dst_buff + dst_origin[0] + dst_row_pitch * dst_origin[1]
        + dst_slice_pitch * dst_origin[2];

    size_t j, k;

    /* TODO: handle overlaping regions */
    if ((src_row_pitch == dst_row_pitch && dst_row_pitch == region[0])
                && (src_slice_pitch == dst_slice_pitch
                && dst_slice_pitch == (region[1] * region[0]))) {
        memcpy (adjusted_dst_ptr, adjusted_src_ptr,
                region[2] * region[1] * region[0]);
    } else {
        for (k = 0; k < region[2]; ++k)
            for (j = 0; j < region[1]; ++j)
                memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
                        adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
                        region[0]);
    }

    err = vt_copy_to_dev(d->vt_device, dev_dst_addr, dst_buff, dst_buf->size, 0, 0);
    assert(0 == err);

    free(src_buff);
    free(dst_buff);
}

void
pocl_ventus_memfill (void *data, pocl_mem_identifier *dst_mem_id,
                     cl_mem dst_buf, size_t size, size_t offset,
                     const void *__restrict__ pattern, size_t pattern_size)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;
  void *host_ptr = pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, size);
  assert(host_ptr);
  uint64_t dev_addr = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr) + offset;
  pocl_fill_aligned_buf_with_pattern (host_ptr, 0, size, pattern, pattern_size);
  int err = vt_copy_to_dev(d->vt_device, dev_addr, host_ptr, size, 0, 0);
  #ifdef PRINT_CHISEL_TESTCODE
  g_vt_dump_mem.emplace_back(dev_addr, size);
  g_vt_dump_mem.back().data.assign((uint8_t*)host_ptr, (uint8_t*)host_ptr + size);
  #endif
  assert(0 == err);
  POCL_MEM_FREE(host_ptr);
}

cl_int
pocl_ventus_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                                 cl_mem src_buf, mem_mapping_t *map)
{
    struct vt_device_data_t *d = (struct vt_device_data_t *)data;
    uint64_t dev_addr = reinterpret_cast<uint64_t>(src_mem_id->mem_ptr) + map->offset;

    assert (map->host_ptr);

    if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION) {
        return CL_SUCCESS;
    }

    int err = vt_copy_from_dev(d->vt_device, dev_addr, map->host_ptr, map->size, 0, 0);
    if (err) {
        return CL_MAP_FAILURE;
    }

    return CL_SUCCESS;
}

cl_int
pocl_ventus_unmap_mem (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, mem_mapping_t *map)
{
    struct vt_device_data_t *d = (struct vt_device_data_t *)data;
    uint64_t dev_addr = reinterpret_cast<uint64_t>(dst_mem_id->mem_ptr) + map->offset;
    assert (map->host_ptr);

    if (map->map_flags != CL_MAP_READ) {
        int err = vt_copy_to_dev(d->vt_device, dev_addr, map->host_ptr, map->size, 0, 0);
        if (err) {
            return CL_MAP_FAILURE;
        }
    }

    return CL_SUCCESS;
}

int pocl_ventus_setup_metadata  (cl_device_id device, cl_program program,
                                 unsigned program_device_i)
{
    return pocl_driver_setup_metadata  (device, program,
     program_device_i);
}

int pocl_ventus_build_program(cl_program program,
                            unsigned device_i,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  return 0;//unused file now.
}

int pocl_ventus_build_source (cl_program program, cl_uint device_i,
                              cl_uint num_input_headers,
                              const cl_program *input_headers,
                              const char **header_include_names,
                              int link_builtin_lib) {
    return pocl_driver_build_source(program, device_i, num_input_headers,
                                       input_headers, header_include_names, 0);
}

int pocl_ventus_post_build_program (cl_program program, cl_uint device_i) {
  std::string clang_path(CLANG);
	if (!pocl_exists(clang_path.c_str())) {
    // Using VENTUS_INSTALL_PREFIX environment to get other clang_path
    std::string ventus_install_prefix(VENTUS_INSTALL_PREFIX_DIR);
    std::string clang_install_path = ventus_install_prefix + "/bin/clang";
    clang_path = clang_install_path;
    if(!pocl_exists(clang_install_path .c_str())) {
      POCL_MSG_ERR("$CLANG: '%s' or '%s' doesn't exist\n", clang_path.c_str(),
                                          clang_install_path.c_str() );
      return -1;
    }
	}
  std::stringstream ss_cmd;
	std::stringstream ss_out;

  char program_bc_path[POCL_FILENAME_LENGTH];

  static uint counter = 0;
  program_ids.insert(std::pair<uint64_t, uint>(uint64_t(program), counter));
  counter++;


  //pocl_cache_create_program_cachedir(program, device_i, program->source,
  //                                     strlen(program->source),
  //                                     program_bc_path);
  //TODO: move .cl and .riscv file into program_bc_path, and let spike read file from this path.
  char filename[256] = "object";
  std::string id = std::to_string(program_ids[uint64_t(program)]);
  strcat(filename, id.c_str());
  char cl_filename[256];
  strcpy(cl_filename, filename);
  strcat(cl_filename, ".cl");
  std::ofstream outfile(cl_filename);
  outfile << program->source;
  outfile.close();

  cl_device_id device = program->devices[device_i];

    ss_cmd << clang_path <<" -cl-std=CL2.0 " << "-target " << device->llvm_target_triplet << " -mcpu=" << device->llvm_cpu  << " " << cl_filename << "  " << " -o " << filename << ".riscv ";
	for(int i = 0; ventus_final_ld_flags[i] != NULL; i++) {
		ss_cmd << ventus_final_ld_flags[i];
	}
	for(int i = 0; ventus_other_compile_flags[i] != NULL; i++) {
		ss_cmd << ventus_other_compile_flags[i];
	}
  if(program->kernel_meta) {
    ss_cmd << "-Wl,--init=" << program->kernel_meta->name << " ";
  }
#ifdef POCL_DEBUG_FLAG_GENERAL
	ss_cmd << " -w ";
#endif
  ss_cmd << " -D__opencl_c_generic_address_space=1 -D__opencl_c_named_address_space_builtins=1 ";
  ss_cmd << " -D__OPENCL_VERSION__=" << device->version_as_int << " ";
	ss_cmd << program->compiler_options << std::endl;
	POCL_MSG_PRINT_VENTUS("running \"%s\"\n", ss_cmd.str().c_str());
  SPDLOG_INFO("running compiler \"{}\"\n", ss_cmd.str());

	FILE *fp = popen(ss_cmd.str().c_str(), "r");
	if(fp == NULL) {
		POCL_MSG_ERR("running compile kernel failed");
		return -1;
	}
	char temp[1024];

	while (fgets(temp, 1024, fp) != NULL)
	{
		ss_out << temp;
	}
	int status=pclose(fp);
    if (status == -1) {
        perror("pclose() failed");
        exit(EXIT_FAILURE);
    } else {
        POCL_MSG_PRINT_VENTUS("after calling clang, the output is : \"%s\"\n", ss_out.str().c_str());
    }


    /*const char* env_var = std::getenv("POCL_PRINT_CHISEL_TESTCODE");
    if (env_var != nullptr) {
        if (std::string(env_var) == "y") {
            // 使用宏定义1
            #ifndef PRINT_CHISEL_TESTCODE
            #define PRINT_CHISEL_TESTCODE
            #endif
            POCL_MSG_PRINT_VENTUS("generate chisel testcode\n");
        }
    } */


  pocl_ventus_release_IR(program);
return 0;

}
