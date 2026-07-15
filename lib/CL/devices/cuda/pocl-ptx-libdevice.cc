/* CUDA libdevice discovery, linking, and OpenCL call mapping. */

#include "config.h"

#include "LLVMUtils.h"
#include "pocl-ptx-gen-internal.h"
#include "pocl-ptx-gen.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct LibDeviceSearchPath {
  const char *Base;
  const char *NVVM;
  int LegacySM;
};

static int formatLibDevicePath(char Path[PATH_MAX],
                               const LibDeviceSearchPath &Search, bool Legacy) {
  static const char *CurrentFormat = "%s%s/libdevice/libdevice.10.bc";
  static const char *LegacyFormat = "%s%s/libdevice/libdevice.compute_%d.10.bc";
  int Length = Legacy ? snprintf(Path, PATH_MAX, LegacyFormat, Search.Base,
                                 Search.NVVM, Search.LegacySM)
                      : snprintf(Path, PATH_MAX, CurrentFormat, Search.Base,
                                 Search.NVVM);
  if (Length >= 0 && static_cast<size_t>(Length) < PATH_MAX)
    return 0;
  POCL_MSG_ERR("[CUDA] libdevice search path exceeds PATH_MAX\n");
  return -1;
}

} // namespace

int findLibDevice(char LibDevicePath[PATH_MAX], const char *Arch) {
  static constexpr unsigned SM20 = 20;
  static constexpr unsigned SM30 = 30;
  static constexpr unsigned SM35 = 35;
  static constexpr unsigned SM37 = 37;
  static constexpr unsigned SM50 = 50;
  static constexpr unsigned SM53 = 53;

  if (!Arch || strncmp(Arch, "sm_", 3) != 0) {
    POCL_MSG_ERR("[CUDA] invalid GPU architecture\n");
    return 1;
  }
  char *End;
  unsigned long SM = strtoul(Arch + 3, &End, 10);
  if (!SM || strlen(End)) {
    POCL_MSG_ERR("[CUDA] invalid GPU architecture %s\n", Arch);
    return 1;
  }

  int LibDeviceSM = SM30;
  if (SM < SM30 || (SM > SM30 && SM < SM35))
    LibDeviceSM = SM20;
  else if (SM == SM30 || (SM > SM37 && SM < SM50))
    LibDeviceSM = SM30;
  else if (SM <= SM37)
    LibDeviceSM = SM35;
  else if (SM <= SM53)
    LibDeviceSM = SM50;

  const char *BasePaths[] = {
      pocl_get_string_option("POCL_CUDA_TOOLKIT_PATH", CUDA_TOOLKIT_ROOT_DIR),
      pocl_get_string_option("CUDA_HOME", "/usr/local/cuda"),
      "/usr/local/lib/cuda",
      "/usr/local/lib",
      "/usr/lib",
  };
  static const char *NVVMPaths[] = {"/nvvm", "/nvidia-cuda-toolkit", ""};

  for (const char *BasePath : BasePaths)
    for (const char *NVVMPath : NVVMPaths) {
      const LibDeviceSearchPath Search = {BasePath, NVVMPath, LibDeviceSM};
      if (formatLibDevicePath(LibDevicePath, Search, false) != 0)
        return 1;
      POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                      "looking for libdevice at '%s'\n", LibDevicePath);
      if (pocl_exists(LibDevicePath)) {
        POCL_MSG_PRINT_CUDA("found libdevice at '%s'\n", LibDevicePath);
        return 0;
      }

      if (formatLibDevicePath(LibDevicePath, Search, true) != 0)
        return 1;
      POCL_MSG_PRINT2(CUDA, __FUNCTION__, __LINE__,
                      "looking for libdevice at '%s'\n", LibDevicePath);
      if (pocl_exists(LibDevicePath)) {
        POCL_MSG_PRINT_CUDA("found libdevice at '%s'\n", LibDevicePath);
        return 0;
      }
    }
  return 1;
}

namespace pocl {
namespace cuda {

int linkLibDevice(llvm::Module *Module, const char *LibDevicePath,
                  bool FlushDenorms) {
  auto Buffer = llvm::MemoryBuffer::getFile(LibDevicePath);
  if (!Buffer) {
    POCL_MSG_ERR("[CUDA] failed to open libdevice library file\n");
    return -1;
  }

  POCL_MSG_PRINT_INFO("loading libdevice from '%s'\n", LibDevicePath);
  auto LibDeviceModule =
      parseBitcodeFile(Buffer->get()->getMemBufferRef(), Module->getContext());
  if (auto Error = LibDeviceModule.takeError()) {
    POCL_MSG_ERR("[CUDA] failed to load libdevice bitcode:\n%s\n",
                 toString(std::move(Error)).c_str());
    return -1;
  }

  (*LibDeviceModule)->setTargetTriple(Module->getTargetTriple());
  (*LibDeviceModule)->setDataLayout(Module->getDataLayout());
  llvm::Linker Linker(*Module);
  if (Linker.linkInModule(std::move(LibDeviceModule.get()),
                          llvm::Linker::Flags::LinkOnlyNeeded)) {
    POCL_MSG_ERR("[CUDA] failed to link to libdevice\n");
    return -1;
  }

  llvm::LLVMContext &Context = Module->getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Context);
  llvm::Metadata *Four =
      llvm::ValueAsMetadata::get(llvm::ConstantInt::getSigned(I32, 4));
  llvm::Metadata *Name = llvm::MDString::get(Context, "nvvm-reflect-ftz");
  llvm::Metadata *Ftz = llvm::ConstantAsMetadata::get(
      llvm::ConstantInt::getSigned(I32, FlushDenorms ? 1 : 0));
  Module->addModuleFlag(llvm::MDNode::get(Context, {Four, Name, Ftz}));

  auto PreserveKernel = [](const llvm::GlobalValue &Value) {
    const auto *Function = llvm::dyn_cast<llvm::Function>(&Value);
    return Function && pocl::isKernelToProcess(*Function);
  };
  internalizeModule(*Module, PreserveKernel);
  populateModulePM(nullptr, Module, 3);
  return 0;
}

struct FunctionMapEntry {
  const char *OpenCLName;
  const char *LibDeviceName;
};

static const FunctionMapEntry FunctionMap[] = {
// clang-format off
#define LDMAP(name)                                                            \
  {        name "f",    "__nv_" name "f"},                                   \
  {        name,        "__nv_" name},                                        \
  {"llvm." name ".f32", "__nv_" name "f"},                                 \
  {"llvm." name ".f64", "__nv_" name},
  LDMAP ("acos")
  LDMAP ("acosh")
  LDMAP ("asin")
  LDMAP ("asinh")
  LDMAP ("atan")
  LDMAP ("atanh")
  LDMAP ("atan2")
  LDMAP ("cbrt")
  LDMAP ("ceil")
  LDMAP ("copysign")
  LDMAP ("cos")
  LDMAP ("cosh")
  LDMAP ("exp")
  LDMAP ("exp2")
  LDMAP ("expm1")
  LDMAP ("fdim")
  LDMAP ("floor")
  LDMAP ("fmax")
  LDMAP ("fmin")
  LDMAP ("hypot")
  LDMAP ("ilogb")
  LDMAP ("lgamma")
  LDMAP ("log")
  LDMAP ("log2")
  LDMAP ("log10")
  LDMAP ("log1p")
  LDMAP ("logb")
  LDMAP ("nextafter")
  LDMAP ("remainder")
  LDMAP ("rint")
  LDMAP ("round")
  LDMAP ("sin")
  LDMAP ("sinh")
  LDMAP ("sqrt")
  LDMAP ("tan")
  LDMAP ("tanh")
  LDMAP ("trunc")
#undef LDMAP
  { "llvm.copysign.f32", "__nv_copysignf" },
  { "llvm.copysign.f64", "__nv_copysign" },
  { "llvm.pow.f32", "__nv_powf" },
  { "llvm.pow.f64", "__nv_pow" },
  { "llvm.powi.f32", "__nv_powif" },
  { "llvm.powi.f64", "__nv_powi" },
  { "frexp", "__nv_frexp" },
  { "frexpf", "__nv_frexpf" },
  { "llvm.frexp.f64.i32", "frexp_f64_i32" },
  { "llvm.frexp.f32.i32", "frexpf_f32_i32" },
  { "tgamma", "__nv_tgamma" },
  { "tgammaf", "__nv_tgammaf" },
  { "ldexp", "__nv_ldexp" },
  { "ldexpf", "__nv_ldexpf" },
  { "llvm.ldexp.f64.i32", "__nv_ldexp" },
  { "llvm.ldexp.f32.i32", "__nv_ldexpf" },
  { "modf", "__nv_modf" },
  { "modff", "__nv_modff" },
  { "remquo", "__nv_remquo" },
  { "remquof", "__nv_remquof" },
  { "erf", "__nv_erf" },
  { "erff", "__nv_erff" },
    // clang-format on
};

static void replaceCalls(llvm::Module *Module, const FunctionMapEntry &Entry) {
  llvm::Function *Function = Module->getFunction(Entry.OpenCLName);
  if (!Function)
    return;

  std::vector<llvm::Value *> Users(Function->user_begin(),
                                   Function->user_end());
  for (llvm::Value *User : Users) {
    auto *Call = llvm::dyn_cast<llvm::CallInst>(User);
    if (!Call)
      continue;
    llvm::FunctionCallee Callee = Module->getOrInsertFunction(
        Entry.LibDeviceName, Function->getFunctionType());
    auto *Replacement = llvm::CallInst::Create(
        llvm::cast<llvm::Function>(Callee.getCallee()),
        std::vector<llvm::Value *>(Call->arg_begin(), Call->arg_end()), "",
        Call);
    Replacement->takeName(Call);
    Call->replaceAllUsesWith(Replacement);
    Call->eraseFromParent();
  }
  Function->eraseFromParent();
}

void mapLibDeviceCalls(llvm::Module *Module) {
  for (const FunctionMapEntry &Entry : FunctionMap)
    replaceCalls(Module, Entry);
}

} // namespace cuda
} // namespace pocl
