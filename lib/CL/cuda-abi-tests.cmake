# Host-only tests link the normal core objects to access hidden utility helpers.
# No public OpenCL/CUDA query or model/device entry is invoked by either test.
foreach(test_name IN ITEMS kernel_wg_info cuda_launch_arguments)
  string(REPLACE "_" "-" test_source "${test_name}")
  add_executable(test_${test_name}
    test-${test_source}.c ${LIBPOCL_OBJS})
  target_include_directories(test_${test_name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/devices/cuda" ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})
  target_link_libraries(test_${test_name} PRIVATE ${POCL_PRIVATE_LINK_LIST})
  add_test(NAME unit/test_${test_name} COMMAND test_${test_name})
  set_tests_properties(unit/test_${test_name} PROPERTIES
    LABELS "cuda;offline" TIMEOUT 60)
endforeach()
