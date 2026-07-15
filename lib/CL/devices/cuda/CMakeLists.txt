#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

find_package(CUDA REQUIRED)
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

# this is required, because we're compiling the C++ files of this directory
# with LLVM-provided flags, which usually contain their own setting of
# C++ standard, which may be in conflict with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
set(CUDA_PTX_SOURCES
  pocl-ptx-alignment.cc
  pocl-ptx-gen.cc
  pocl-ptx-gen.h
  pocl-ptx-gen-internal.h
  pocl-ptx-kernel-abi.cc
  pocl-ptx-libdevice.cc
  pocl-ptx-printf.cc
  pocl-ptx-printf.h
  pocl-ptx-rotate.cc
  pocl-ptx-rotate.h
  pocl-ptx-scalarize.cc
  pocl-ptx-scalarize.h
  pocl-ptx-vector-args.cc
  pocl-ptx-vector-args.h)
set(CUDA_DRIVER_FRAGMENTS
  pocl-cuda-device.inc
  pocl-cuda-event.inc
  pocl-cuda-kernel.inc
  pocl-cuda-memory.inc
  pocl-cuda-program.inc
  pocl-cuda-submit.inc
  pocl-cuda-svm.inc
  pocl-cuda-target.inc
  pocl-cuda-transfer.inc)
set_source_files_properties(${CUDA_DRIVER_FRAGMENTS}
  PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${CUDA_PTX_SOURCES}
  PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
include_directories(${LLVM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_pocl_device_library("pocl-devices-cuda"
  pocl-cuda.c
  pocl-cuda.h
  pocl-cuda-queue.c
  pocl-cuda-queue.h
  ${CUDA_DRIVER_FRAGMENTS}
  ${CUDA_PTX_SOURCES})
target_compile_definitions("pocl-devices-cuda" PRIVATE "-DCUDA_TOOLKIT_ROOT_DIR=\"${CUDA_TOOLKIT_ROOT_DIR}\"")

if(ENABLE_LOADABLE_DRIVERS)
  if(LLVM_PACKAGE_VERSION)
    set(CUDA_PRIVATE_LINK_LIST ${POCL_CLANG_LINK_TARGETS} ${POCL_LLVM_LINK_TARGETS})
  else()
    set(CUDA_PRIVATE_LINK_LIST ${CLANG_LINK_LIBRARIES} ${LLVM_LINK_LIBRARIES} ${LLVM_SYSLIBS})
  endif()
  if(CLANG_LINK_DIRS AND (NOT CLANG_LINK_DIRS STREQUAL LLVM_LIBDIR))
    target_link_directories(pocl-devices-cuda PRIVATE ${CLANG_LINK_DIRS})
  endif()
  if(LLVM_LINK_DIRS AND (NOT LLVM_LINK_DIRS STREQUAL LLVM_LIBDIR))
    target_link_directories(pocl-devices-cuda PRIVATE ${LLVM_LINK_DIRS})
  endif()
  target_link_libraries(pocl-devices-cuda PRIVATE ${CUDA_POCL_TARGETS} ${CUDA_PRIVATE_LINK_LIST} ${PTHREAD_LIBRARY})
endif()

if(ENABLE_TESTS)
  add_executable(test_cuda_rotate
    test-ptx-rotate.cc
    pocl-ptx-rotate.cc)
  target_link_libraries(test_cuda_rotate
    PRIVATE ${POCL_LIBRARY_NAME} ${POCL_LLVM_LINK_TARGETS}
            ${LLVM_LINK_LIBRARIES})
  add_test(NAME unit/test_cuda_rotate COMMAND test_cuda_rotate)
  set_tests_properties(unit/test_cuda_rotate
    PROPERTIES LABELS "cuda;offline")
  add_executable(test_cuda_scalarize
    test-ptx-scalarize.cc
    pocl-ptx-scalarize.cc)
  target_link_libraries(test_cuda_scalarize
    PRIVATE ${POCL_LIBRARY_NAME} ${POCL_LLVM_LINK_TARGETS}
            ${LLVM_LINK_LIBRARIES})
  add_test(NAME unit/test_cuda_scalarize COMMAND test_cuda_scalarize)
  set_tests_properties(unit/test_cuda_scalarize
    PROPERTIES LABELS "cuda;offline")
  add_executable(test_cuda_printf
    test-ptx-printf.cc
    pocl-ptx-printf.cc)
  target_link_libraries(test_cuda_printf
    PRIVATE ${POCL_LIBRARY_NAME} ${POCL_LLVM_LINK_TARGETS}
            ${LLVM_LINK_LIBRARIES})
  add_test(NAME unit/test_cuda_printf COMMAND test_cuda_printf)
  set_tests_properties(unit/test_cuda_printf
    PROPERTIES LABELS "cuda;offline")
  add_executable(test_cuda_vector_args
    test-ptx-vector-args.cc
    pocl-ptx-vector-args.cc)
  target_link_libraries(test_cuda_vector_args
    PRIVATE ${POCL_LIBRARY_NAME} ${POCL_LLVM_LINK_TARGETS}
            ${LLVM_LINK_LIBRARIES})
  add_test(NAME unit/test_cuda_vector_args COMMAND test_cuda_vector_args)
  set_tests_properties(unit/test_cuda_vector_args
    PROPERTIES LABELS "cuda;offline")
  add_test(NAME unit/test_cuda_cts_runner
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tools/scripts/cuda_stable/test_run_cts_quick.py")
  set_tests_properties(unit/test_cuda_cts_runner
    PROPERTIES LABELS "cuda;offline")
endif()


install(FILES "builtins.cl" "builtins_sm50.ptx" "builtins_sm70.ptx"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR_REL}/cuda" COMPONENT "lib")
