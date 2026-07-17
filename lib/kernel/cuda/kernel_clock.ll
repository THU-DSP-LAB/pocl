; CUDA implements all OpenCL kernel clock scopes with the device-wide
; monotonically increasing PTX global timer. The side effect and memory
; clobber preserve the ordering of consecutive reads.

define internal i64 @pocl_cuda_read_globaltimer() alwaysinline nounwind {
entry:
  %clock = call i64 asm sideeffect "mov.u64 $0, %globaltimer;", "=l,~{memory}"()
  ret i64 %clock
}

define i64 @_Z21_cl_clock_read_devicev() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  ret i64 %clock
}

define i64 @_Z25_cl_clock_read_work_groupv() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  ret i64 %clock
}

define i64 @_Z24_cl_clock_read_sub_groupv() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  ret i64 %clock
}

define <2 x i32> @_Z26_cl_clock_read_hilo_devicev() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  %lo = trunc i64 %clock to i32
  %shifted = lshr i64 %clock, 32
  %hi = trunc i64 %shifted to i32
  %with_lo = insertelement <2 x i32> poison, i32 %lo, i32 0
  %result = insertelement <2 x i32> %with_lo, i32 %hi, i32 1
  ret <2 x i32> %result
}

define <2 x i32> @_Z30_cl_clock_read_hilo_work_groupv() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  %lo = trunc i64 %clock to i32
  %shifted = lshr i64 %clock, 32
  %hi = trunc i64 %shifted to i32
  %with_lo = insertelement <2 x i32> poison, i32 %lo, i32 0
  %result = insertelement <2 x i32> %with_lo, i32 %hi, i32 1
  ret <2 x i32> %result
}

define <2 x i32> @_Z29_cl_clock_read_hilo_sub_groupv() nounwind {
entry:
  %clock = call i64 @pocl_cuda_read_globaltimer()
  %lo = trunc i64 %clock to i32
  %shifted = lshr i64 %clock, 32
  %hi = trunc i64 %shifted to i32
  %with_lo = insertelement <2 x i32> poison, i32 %lo, i32 0
  %result = insertelement <2 x i32> %with_lo, i32 %hi, i32 1
  ret <2 x i32> %result
}
