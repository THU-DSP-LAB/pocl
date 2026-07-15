/* CUDA kernel argument alignment metadata and subgroup constants. */

#include "config.h"

#include "LLVMUtils.h"
#include "common.h"
#include "pocl-ptx-gen-internal.h"
#include "pocl-ptx-gen.h"
#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace pocl {
namespace cuda {

static size_t inferPointerAlignment(llvm::Argument &Argument) {
  for (llvm::User *User : Argument.users()) {
    auto *Address = llvm::dyn_cast<llvm::GetElementPtrInst>(User);
    if (!Address)
      continue;
    for (llvm::User *AddressUser : Address->users()) {
      if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(AddressUser))
        return Store->getAlign().value();
      if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(AddressUser))
        return Load->getAlign().value();
    }
  }
  return MAX_EXTENDED_ALIGNMENT;
}

static int getArgumentAlignments(llvm::Module *Module, llvm::Function *Kernel,
                                 std::vector<size_t> &Alignments) {
  if (!Module || !Kernel) {
    POCL_MSG_ERR("[CUDA] kernel alignment input is missing\n");
    return -1;
  }

  const llvm::DataLayout &Layout = Module->getDataLayout();
  for (auto &Argument : Kernel->args()) {
    unsigned Index = Argument.getArgNo();
    llvm::Type *Type = Argument.getType();
    Alignments.push_back(0);
    assert(Index < Alignments.size());
    Alignments[Index] = Type->isPointerTy() ? inferPointerAlignment(Argument)
                                            : Layout.getTypeAllocSize(Type);
  }
  return 0;
}

void createAlignmentMap(llvm::Module *Module, AlignmentMap *Map) {
  for (auto &Function : Module->functions()) {
    if (!pocl::isKernelToProcess(Function))
      continue;
    std::string Name = Function.getName().str();
    if (Map->find(Name) != Map->end())
      continue;
    std::vector<size_t> &Alignments = (*Map)[Name];
    Alignments.reserve(4);
    if (getArgumentAlignments(Module, &Function, Alignments) != 0)
      POCL_MSG_ERR("can't determine alignments for kernel %s", Name.c_str());
  }
}

} // namespace cuda
} // namespace pocl

int pocl_cuda_create_alignments(void *LLVMModule, void *Program,
                                void **AlignmentMapPtr) {
  auto *Module = static_cast<llvm::Module *>(LLVMModule);
  if (!Module || !Program || !AlignmentMapPtr || *AlignmentMapPtr) {
    POCL_MSG_ERR("[CUDA] invalid alignment map creation input\n");
    return -1;
  }

  cl_program OpenCLProgram = static_cast<cl_program>(Program);
  auto *ContextData = static_cast<PoclLLVMContextData *>(
      OpenCLProgram->context->llvm_context_data);
  PoclCompilerMutexGuard LockHolder(&ContextData->Lock);

  auto *Map = new pocl::cuda::AlignmentMap;
  pocl::cuda::createAlignmentMap(Module, Map);
  if (pocl_get_bool_option("POCL_LLVM_VERIFY", LLVM_VERIFY_MODULE_DEFAULT) &&
      !pocl::cuda::verifyModule(Module, "getAlignmentMap")) {
    delete Map;
    return -1;
  }
  *AlignmentMapPtr = Map;
  return 0;
}

void pocl_cuda_destroy_alignments(void *LLVMModule, void *AlignmentMapPtr) {
  (void)LLVMModule;
  delete static_cast<pocl::cuda::AlignmentMap *>(AlignmentMapPtr);
}

int pocl_cuda_get_ptr_arg_alignment(const pocl_cuda_alignment_query_t *Query) {
  if (!Query)
    return 1;

  auto *Map = static_cast<pocl::cuda::AlignmentMap *>(Query->alignment_map);
  if (!Map || !Query->kernel_name || !Query->alignments)
    return 1;

  auto Entry = Map->find(Query->kernel_name);
  if (Entry == Map->end()) {
    POCL_MSG_ERR("CUDA alignment map has no kernel named %s\n",
                 Query->kernel_name);
    return 1;
  }
  const std::vector<size_t> &Values = Entry->second;
  std::memcpy(Query->alignments, Values.data(), sizeof(size_t) * Values.size());
  return 0;
}

int pocl_cuda_define_sub_group_size(void *LLVMModule, void *Program,
                                    int SubgroupSize) {
  auto *Module = static_cast<llvm::Module *>(LLVMModule);
  if (!Module || !Program)
    return CL_INVALID_VALUE;

  cl_program OpenCLProgram = static_cast<cl_program>(Program);
  auto *ContextData = static_cast<PoclLLVMContextData *>(
      OpenCLProgram->context->llvm_context_data);
  PoclCompilerMutexGuard LockHolder(&ContextData->Lock);
  llvm::GlobalVariable *SizeVariable =
      Module->getGlobalVariable("_pocl_warp_size");
  if (SizeVariable)
    SizeVariable->setInitializer(
        llvm::ConstantInt::get(SizeVariable->getValueType(), SubgroupSize));
  return CL_SUCCESS;
}
