/* SPDX-License-Identifier: MIT */

__attribute__ ((convergent, overloadable))
void
async_work_group_copy_fence (cl_mem_fence_flags flags)
{
  // The CUDA copy builtins execute each work-item's share synchronously.
  // All work-items complete their preceding shares before starting the next.
  barrier (flags);
}
