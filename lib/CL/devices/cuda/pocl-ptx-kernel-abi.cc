/* CUDA kernel ABI normalization for PTX generation. */

#include "config.h"

#include "LLVMUtils.h"
#include "pocl-ptx-gen-internal.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace pocl {
namespace cuda {

void addKernelAnnotations(llvm::Module *Module) {
  llvm::LLVMContext &Context = Module->getContext();
  llvm::Constant *One =
      llvm::ConstantInt::getSigned(llvm::Type::getInt32Ty(Context), 1);

  auto *Annotations = Module->getNamedMetadata("nvvm.annotations");
  if (Annotations)
    Annotations->eraseFromParent();
  Annotations = Module->getOrInsertNamedMetadata("nvvm.annotations");

  for (auto &Function : Module->functions()) {
    if (!pocl::isKernelToProcess(Function))
      continue;

    llvm::Metadata *FunctionMetadata = llvm::ValueAsMetadata::get(&Function);
    llvm::Metadata *NameMetadata = llvm::MDString::get(Context, "kernel");
    llvm::Metadata *OneMetadata = llvm::ConstantAsMetadata::get(One);
    llvm::MDNode *Node = llvm::MDNode::get(
        Context, {FunctionMetadata, NameMetadata, OneMetadata});
    Annotations->addOperand(Node);
  }
}

static void replaceScalarGlobalInFunction(llvm::Module *Module,
                                          llvm::Function *Function,
                                          const char *Name,
                                          llvm::Value *NewValue) {
  auto *Global = Module->getGlobalVariable(Name);
  if (!Global)
    return;

  std::vector<llvm::Value *> Users(Global->user_begin(), Global->user_end());
  for (llvm::Value *User : Users) {
    auto *Load = llvm::dyn_cast<llvm::LoadInst>(User);
    assert(Load && "Use of a scalar global variable is not a load");
    if (Load->getFunction() != Function)
      continue;
    Load->replaceAllUsesWith(NewValue);
    Load->eraseFromParent();
  }
}

static void
replaceGlobalsWithArguments(llvm::Module *Module,
                            const std::vector<std::string> &Names,
                            const std::vector<llvm::Type *> &Types) {
  assert(Names.size() == Types.size());

  llvm::SmallVector<llvm::Function *, 8> Functions;
  for (auto &Function : Module->functions())
    if (pocl::isKernelToProcess(Function))
      Functions.push_back(&Function);

  for (llvm::Function *Function : Functions) {
    llvm::FunctionType *OldType = Function->getFunctionType();
    std::vector<llvm::Type *> ArgumentTypes(OldType->param_begin(),
                                            OldType->param_end());
    ArgumentTypes.insert(ArgumentTypes.end(), Types.begin(), Types.end());
    llvm::FunctionType *NewType = llvm::FunctionType::get(
        Function->getReturnType(), ArgumentTypes, false);
    llvm::Function *NewFunction = llvm::Function::Create(
        NewType, Function->getLinkage(), Function->getName(), Module);
    NewFunction->takeName(Function);

    llvm::ValueToValueMapTy ValueMap;
    auto NewArgument = NewFunction->arg_begin();
    for (auto &OldArgument : Function->args()) {
      NewArgument->takeName(&OldArgument);
      ValueMap[&OldArgument] = &*NewArgument;
      ++NewArgument;
    }

    llvm::SmallVector<llvm::ReturnInst *, 1> Returns;
    CloneFunctionIntoAbs(NewFunction, Function, ValueMap, Returns);
    for (const std::string &Name : Names) {
      NewArgument->setName(Name);
      replaceScalarGlobalInFunction(Module, NewFunction, Name.c_str(),
                                    &*NewArgument);
      ++NewArgument;
    }
  }

  for (llvm::Function *Function : Functions)
    Function->eraseFromParent();
  for (const std::string &Name : Names) {
    auto *Global = Module->getGlobalVariable(Name);
    if (Global)
      Global->eraseFromParent();
  }
}

void handleGetWorkDim(llvm::Module *Module) {
  llvm::Type *Type = llvm::Type::getInt32Ty(Module->getContext());
  replaceGlobalsWithArguments(Module, {"_work_dim"}, {Type});
}

void handleGetGlobalOffset(llvm::Module *Module) {
  llvm::Type *Type = llvm::Type::getInt32Ty(Module->getContext());
  replaceGlobalsWithArguments(
      Module, {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
      std::vector<llvm::Type *>(3, Type));
}

void handleNonUniformWorkGroup(llvm::Module *Module) {
  llvm::Type *Type = llvm::Type::getInt32Ty(Module->getContext());
  replaceGlobalsWithArguments(
      Module,
      {"_global_size_x", "_global_size_y", "_global_size_z",
       "_num_groups_x", "_num_groups_y", "_num_groups_z",
       "_enqueued_local_size_x", "_enqueued_local_size_y",
       "_enqueued_local_size_z", "_group_offset_x", "_group_offset_y",
       "_group_offset_z"},
      std::vector<llvm::Type *>(12, Type));
}

struct OffsetArgumentRewrite {
  std::vector<llvm::Argument *> Arguments;
  std::vector<llvm::Type *> Types;
  llvm::ValueToValueMapTy ValueMap;
  std::vector<std::pair<llvm::Instruction *, llvm::Instruction *>> Insertions;
  bool HasOffsets = false;
};

struct MemoryRegion {
  llvm::Module *Module;
  unsigned AddressSpace;
  llvm::GlobalVariable *Base;
};

struct MemoryRegionSpec {
  const char *Name;
  unsigned AddressSpace;
  size_t Size;
};

static void addOffsetArgument(llvm::Argument &Argument,
                              const MemoryRegion &Region,
                              OffsetArgumentRewrite &Rewrite) {
  llvm::Type *I32 = llvm::Type::getInt32Ty(Region.Module->getContext());
  auto *Offset = new llvm::Argument(I32, Argument.getName() + "_offset");
  Rewrite.Arguments.push_back(Offset);
  Rewrite.Types.push_back(I32);

  llvm::Value *Zero = llvm::ConstantInt::getSigned(I32, 0);
  auto *Address = llvm::GetElementPtrInst::Create(Region.Base->getValueType(),
                                                  Region.Base, {Zero, Offset});
  auto *Cast = new llvm::BitCastInst(Address, Argument.getType());
  Rewrite.Insertions.push_back({Address, Cast});
  Rewrite.ValueMap[&Argument] = Cast;
  Rewrite.HasOffsets = true;
}

static void buildOffsetRewrite(llvm::Function *Function,
                               const MemoryRegion &Region,
                               OffsetArgumentRewrite &Rewrite) {
  for (auto &Argument : Function->args()) {
    llvm::Type *Type = Argument.getType();
    if (Type->isPointerTy() &&
        Type->getPointerAddressSpace() == Region.AddressSpace)
      addOffsetArgument(Argument, Region, Rewrite);
    else {
      Rewrite.Arguments.push_back(&Argument);
      Rewrite.Types.push_back(Type);
    }
  }
}

static void mapOffsetArguments(llvm::Function *NewFunction,
                               OffsetArgumentRewrite &Rewrite) {
  auto OldArgument = Rewrite.Arguments.begin();
  auto NewArgument = NewFunction->arg_begin();
  for (; NewArgument != NewFunction->arg_end(); ++NewArgument, ++OldArgument) {
    NewArgument->takeName(*OldArgument);
    if ((*OldArgument)->getParent())
      Rewrite.ValueMap[*OldArgument] = &*NewArgument;
    else {
      (*OldArgument)->replaceAllUsesWith(&*NewArgument);
      delete *OldArgument;
    }
  }
}

static bool convertPointerArgumentsToOffsets(llvm::Function *Function,
                                             const MemoryRegion &Region) {
  OffsetArgumentRewrite Rewrite;
  buildOffsetRewrite(Function, Region, Rewrite);
  if (!Rewrite.HasOffsets)
    return false;

  llvm::FunctionType *NewType =
      llvm::FunctionType::get(Function->getReturnType(), Rewrite.Types, false);
  llvm::Function *NewFunction = llvm::Function::Create(
      NewType, Function->getLinkage(), Function->getName(), Region.Module);
  NewFunction->takeName(Function);
  mapOffsetArguments(NewFunction, Rewrite);

  llvm::SmallVector<llvm::ReturnInst *, 1> Returns;
  CloneFunctionIntoAbs(NewFunction, Function, Rewrite.ValueMap, Returns);
  for (auto &Insertion : Rewrite.Insertions) {
    Insertion.first->insertBefore(&*NewFunction->begin()->begin());
    Insertion.second->insertAfter(Insertion.first);
  }
  return true;
}

static llvm::GlobalVariable *createMemoryRegion(llvm::Module *Module,
                                                const MemoryRegionSpec &Spec) {
  llvm::Type *ByteArray = llvm::ArrayType::get(
      llvm::Type::getInt8Ty(Module->getContext()), Spec.Size);
  return new llvm::GlobalVariable(
      *Module, ByteArray, false, llvm::GlobalValue::ExternalLinkage, nullptr,
      Spec.Name, nullptr, llvm::GlobalValue::NotThreadLocal, Spec.AddressSpace,
      false);
}

static void replaceMemoryArguments(const MemoryRegion &Region) {
  llvm::SmallVector<llvm::Function *, 8> OldFunctions;
  for (auto &Function : Region.Module->functions()) {
    if (!pocl::isKernelToProcess(Function))
      continue;
    if (convertPointerArgumentsToOffsets(&Function, Region))
      OldFunctions.push_back(&Function);
  }
  for (llvm::Function *Function : OldFunctions)
    Function->eraseFromParent();
}

void fixConstantMemArgs(llvm::Module *Module) {
  static constexpr size_t ConstantMemorySize = 65536;
  size_t AutomaticSize = 0;
  for (auto &Global : Module->globals())
    if (Global.getType()->getPointerAddressSpace() == 4)
      AutomaticSize += Module->getDataLayout().getTypeAllocSize(
          Global.getInitializer()->getType());

  llvm::GlobalVariable *Base =
      createMemoryRegion(Module, {"_constant_memory_region_", 4,
                                  ConstantMemorySize - AutomaticSize});
  replaceMemoryArguments({Module, 4, Base});
}

void fixLocalMemArgs(llvm::Module *Module) {
  llvm::GlobalVariable *Base =
      createMemoryRegion(Module, {"_shared_memory_region_", 3, 0});
  replaceMemoryArguments({Module, 3, Base});
}

} // namespace cuda
} // namespace pocl
