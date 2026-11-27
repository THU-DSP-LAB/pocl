; GPIDL Driver-SBT flat-address ABI only. This is not the SM89 CUDA ABI.
; Both OpenCL overloads take a generic pointer (NVPTX address space 0).
; Input is a live global, shared (OpenCL local), or thread-private pointer.
; A 32-bit named shared offset is not an input to this function.
target triple = "nvptx64"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

define i32 @_Z9get_fencePKv(ptr %address) alwaysinline nounwind memory(none) {
  %bits = ptrtoint ptr %address to i64
  %tag = lshr i64 %bits, 62
  %global = icmp eq i64 %tag, 0
  %shared = icmp eq i64 %tag, 2
  %nonglobal_flag = zext i1 %shared to i32
  %flag = select i1 %global, i32 2, i32 %nonglobal_flag
  ret i32 %flag
}

define i32 @_Z9get_fencePv(ptr %address) alwaysinline nounwind memory(none) {
  %flag = call i32 @_Z9get_fencePKv(ptr %address)
  ret i32 %flag
}

define ptr addrspace(1) @__to_global(ptr %address) alwaysinline nounwind memory(none) {
  %bits = ptrtoint ptr %address to i64
  %tag = lshr i64 %bits, 62
  %valid = icmp eq i64 %tag, 0
  %named = addrspacecast ptr %address to ptr addrspace(1)
  %result = select i1 %valid, ptr addrspace(1) %named, ptr addrspace(1) null
  ret ptr addrspace(1) %result
}

define ptr addrspace(3) @__to_local(ptr %address) alwaysinline nounwind memory(none) {
  %bits = ptrtoint ptr %address to i64
  %tag = lshr i64 %bits, 62
  %valid = icmp eq i64 %tag, 2
  %named = addrspacecast ptr %address to ptr addrspace(3)
  %result = select i1 %valid, ptr addrspace(3) %named, ptr addrspace(3) null
  ret ptr addrspace(3) %result
}

define ptr @__to_private(ptr %address) alwaysinline nounwind memory(none) {
  %bits = ptrtoint ptr %address to i64
  %tag = lshr i64 %bits, 62
  %valid = icmp eq i64 %tag, 1
  %result = select i1 %valid, ptr %address, ptr null
  ret ptr %result
}
