declare void @llvm.nvvm.barrier0()
declare void @llvm.nvvm.membar.cta()
declare void @llvm.nvvm.membar.gl()
declare void @llvm.nvvm.membar.sys()

; this is merely to enforce non-opaque-pointers
; required because there are no other pointers in the code,
; and llvm-link just sets opaque-pointers setting to whatever default
declare void @__pocl_unused(ptr  %arg1) noduplicate

define void @_Z7barrierj(i32 %flags) noduplicate {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}

define void @_Z22_cl_work_group_barrierj(i32) noduplicate {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}


define void @_Z22_cl_work_group_barrierj12memory_scope(i32, i32) noduplicate {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}

define void @_Z18_cl_read_mem_fencej(i32) noduplicate {
entry:
  call void @llvm.nvvm.membar.cta()
  ret void
}


define void @_Z19_cl_write_mem_fencej(i32) noduplicate {
entry:
  call void @llvm.nvvm.membar.cta()
  ret void
}

define void @_Z13_cl_mem_fencej(i32) noduplicate {
entry:
  call void @llvm.nvvm.membar.cta()
  ret void
}

define void @_Z26_cl_atomic_work_item_fencej12memory_order12memory_scope(i32 %flags, i32 %order, i32 %scope) noduplicate {
entry:
  switch i32 %scope, label %system_scope [
    i32 0, label %exit
    i32 1, label %work_group_scope
    i32 2, label %device_scope
    i32 4, label %work_group_scope
  ]

work_group_scope:
  call void @llvm.nvvm.membar.cta()
  br label %exit

device_scope:
  call void @llvm.nvvm.membar.gl()
  br label %exit

system_scope:
  call void @llvm.nvvm.membar.sys()
  br label %exit

exit:
  ret void
}
