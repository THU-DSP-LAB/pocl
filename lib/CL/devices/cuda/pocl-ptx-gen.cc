/* pocl-ptx-gen.cc - orchestration for CUDA PTX code generation. */

#include "config.h"

#include "pocl-ptx-gen-internal.h"
#include "pocl-ptx-gen.h"
#include "pocl-ptx-printf.h"
#include "pocl-ptx-rotate.h"
#include "pocl-ptx-scalarize.h"
#include "pocl-ptx-vector-args.h"
#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace pocl {
namespace cuda {

bool verifyModule(llvm::Module *Module, const char *Step) {
  std::string Error;
  llvm::raw_string_ostream ErrorStream(Error);
  if (!llvm::verifyModule(*Module, &ErrorStream))
    return true;
  POCL_MSG_ERR("[CUDA] ptx-gen step %s: module verification FAILED\n%s\n", Step,
               Error.c_str());
  return false;
}

} // namespace cuda
} // namespace pocl

namespace {

struct PtxTransformOptions {
  const char *LibDevicePath;
  bool VerifyModule;
  bool FlushDenorms;
};

bool verifyAtStep(llvm::Module *Module, const char *Step, bool Enabled) {
  return !Enabled || pocl::cuda::verifyModule(Module, Step);
}

int applyCoreTransforms(llvm::Module *Module, bool VerifyModule) {
  if (!pocl_cuda_lower_printf(Module) ||
      !verifyAtStep(Module, "lowerPrintf", VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::fixConstantMemArgs(Module);
  if (!verifyAtStep(Module, "fixConstantMemArgs", VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::fixLocalMemArgs(Module);
  if (!verifyAtStep(Module, "fixLocalMemArgs", VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::handleGetWorkDim(Module);
  if (!verifyAtStep(Module, "handleGetWorkDim", VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::handleGetGlobalOffset(Module);
  if (!verifyAtStep(Module, "handleGetGlobalOffset", VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::handleNonUniformWorkGroup(Module);
  return verifyAtStep(Module, "handleNonUniformWorkGroup", VerifyModule)
             ? CL_SUCCESS
             : CL_BUILD_PROGRAM_FAILURE;
}

int applyBackendTransforms(llvm::Module *Module,
                           const PtxTransformOptions &Options) {
  pocl_cuda_fix_vector_args(Module);
  if (!verifyAtStep(Module, "fixVectorArgs", Options.VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl_cuda_scalarize_vectors(Module);
  if (!verifyAtStep(Module, "scalarizeVectors", Options.VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::addKernelAnnotations(Module);
  if (!verifyAtStep(Module, "addAnnotations", Options.VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  pocl::cuda::mapLibDeviceCalls(Module);
  if (!verifyAtStep(Module, "mapLibDeviceCalls", Options.VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  if (pocl::cuda::linkLibDevice(Module, Options.LibDevicePath,
                                Options.FlushDenorms) != 0 ||
      !verifyAtStep(Module, "linkLibDevice", Options.VerifyModule))
    return CL_BUILD_PROGRAM_FAILURE;

  unsigned Count = pocl_cuda_lower_i64_rotates(Module);
  POCL_MSG_PRINT_CUDA("lowered %u i64 rotate implementations\n", Count);
  return verifyAtStep(Module, "lowerI64Rotates", Options.VerifyModule)
             ? CL_SUCCESS
             : CL_BUILD_PROGRAM_FAILURE;
}

int preparePtxModule(llvm::Module *Module, const PtxTransformOptions &Options,
                     void **AlignmentMapPointer) {
  auto AlignmentMap = std::make_unique<pocl::cuda::AlignmentMap>();
  if (!AlignmentMapPointer || *AlignmentMapPointer)
    return CL_INVALID_VALUE;

  pocl::cuda::createAlignmentMap(Module, AlignmentMap.get());
  if (!verifyAtStep(Module, "getAlignmentMap", Options.VerifyModule) ||
      applyCoreTransforms(Module, Options.VerifyModule) != CL_SUCCESS ||
      applyBackendTransforms(Module, Options) != CL_SUCCESS)
    return CL_BUILD_PROGRAM_FAILURE;

  if (pocl_get_bool_option("POCL_CUDA_DUMP_NVVM", 0)) {
    std::string ModuleText;
    llvm::raw_string_ostream Stream(ModuleText);
    Module->print(Stream, nullptr);
    POCL_MSG_PRINT_INFO("NVVM module:\n%s\n", ModuleText.c_str());
  }

  *AlignmentMapPointer = AlignmentMap.release();
  return CL_SUCCESS;
}

} // namespace

int pocl_ptx_gen(const pocl_ptx_gen_options_t *Options) {
  if (!Options || !Options->program || !Options->llvm_module ||
      !Options->ptx_filename || !Options->libdevice_path ||
      !Options->alignment_map) {
    POCL_MSG_ERR("[CUDA] ptx-gen: invalid generation options\n");
    return CL_BUILD_PROGRAM_FAILURE;
  }

  auto *Module = static_cast<llvm::Module *>(Options->llvm_module);
  cl_program Program = static_cast<cl_program>(Options->program);
  auto *ContextData =
      static_cast<PoclLLVMContextData *>(Program->context->llvm_context_data);
  const PtxTransformOptions TransformOptions = {
      Options->libdevice_path,
      pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT) != 0,
      Program->flush_denorms != 0,
  };
  {
    PoclCompilerMutexGuard LockHolder(&ContextData->Lock);
    if (preparePtxModule(Module, TransformOptions, Options->alignment_map) !=
        CL_SUCCESS)
      return CL_BUILD_PROGRAM_FAILURE;
  }

  std::string Features =
      std::string("+ptx") + std::to_string(Options->ptx_version);
  char *Content = nullptr;
  size_t ContentSize = 0;
  int Error = pocl_llvm_codegen(static_cast<cl_device_id>(Options->device),
                                Program, Features.c_str(), Module, CL_TRUE,
                                CL_FALSE, &Content, &ContentSize);
  if (Error != CL_SUCCESS) {
    POCL_MSG_ERR("[CUDA] ptx-gen: failed codegen PTX %s\n",
                 Options->ptx_filename);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  assert(Content && ContentSize);
  int WriteError =
      pocl_write_file(Options->ptx_filename, Content, ContentSize, 0);
  free(Content);
  if (!WriteError)
    return CL_SUCCESS;
  POCL_MSG_ERR("[CUDA] ptx-gen: failed to write final PTX into %s\n",
               Options->ptx_filename);
  return CL_BUILD_PROGRAM_FAILURE;
}
