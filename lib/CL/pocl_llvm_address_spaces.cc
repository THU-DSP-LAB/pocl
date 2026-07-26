#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
POP_COMPILER_DIAGS

#include "pocl_llvm_address_spaces.hh"

namespace {

class AddressSpaceTypeRemapper final : public llvm::ValueMapTypeRemapper {
public:
  AddressSpaceTypeRemapper(unsigned Source, unsigned Target)
      : Source(Source), Target(Target) {}

  llvm::Type *remapType(llvm::Type *Type) override {
    auto Cached = Cache.find(Type);
    if (Cached != Cache.end())
      return Cached->second;

    llvm::Type *Mapped = remapUncached(Type);
    Cache[Type] = Mapped;
    return Mapped;
  }

private:
  llvm::Type *remapUncached(llvm::Type *Type) {
    if (auto *Pointer = llvm::dyn_cast<llvm::PointerType>(Type))
      return remapPointer(Pointer);
    if (auto *Function = llvm::dyn_cast<llvm::FunctionType>(Type))
      return remapFunction(Function);
    if (auto *Array = llvm::dyn_cast<llvm::ArrayType>(Type))
      return llvm::ArrayType::get(remapType(Array->getElementType()),
                                  Array->getNumElements());
    if (auto *Vector = llvm::dyn_cast<llvm::FixedVectorType>(Type))
      return llvm::FixedVectorType::get(remapType(Vector->getElementType()),
                                        Vector->getNumElements());
    if (auto *Vector = llvm::dyn_cast<llvm::ScalableVectorType>(Type))
      return llvm::ScalableVectorType::get(remapType(Vector->getElementType()),
                                           Vector->getMinNumElements());
    if (auto *Structure = llvm::dyn_cast<llvm::StructType>(Type))
      return remapStruct(Structure);
    return Type;
  }

  llvm::Type *remapPointer(llvm::PointerType *Pointer) const {
    if (Pointer->getAddressSpace() != Source)
      return Pointer;
    return llvm::PointerType::get(Pointer->getContext(), Target);
  }

  llvm::Type *remapFunction(llvm::FunctionType *Function) {
    llvm::SmallVector<llvm::Type *, 8> Parameters;
    for (llvm::Type *Parameter : Function->params())
      Parameters.push_back(remapType(Parameter));
    return llvm::FunctionType::get(remapType(Function->getReturnType()),
                                   Parameters, Function->isVarArg());
  }

  llvm::Type *remapStruct(llvm::StructType *Structure) {
    if (Structure->isOpaque())
      return Structure;

    llvm::SmallVector<llvm::Type *, 8> Elements;
    bool Changed = false;
    for (llvm::Type *Element : Structure->elements()) {
      llvm::Type *Mapped = remapType(Element);
      Elements.push_back(Mapped);
      Changed |= Mapped != Element;
    }
    if (!Changed)
      return Structure;
    if (Structure->isLiteral())
      return llvm::StructType::get(Structure->getContext(), Elements,
                                   Structure->isPacked());

    std::string Name = (Structure->getName() + ".as-remapped").str();
    auto *Mapped = llvm::StructType::create(Structure->getContext(), Name);
    Cache[Structure] = Mapped;
    Mapped->setBody(Elements, Structure->isPacked());
    return Mapped;
  }

  const unsigned Source;
  const unsigned Target;
  llvm::DenseMap<llvm::Type *, llvm::Type *> Cache;
};

using FunctionPair = std::pair<llvm::Function *, llvm::Function *>;
using GlobalPair = std::pair<llvm::GlobalVariable *, llvm::GlobalVariable *>;

llvm::GlobalVariable *
createMappedGlobal(llvm::GlobalVariable *Source,
                   AddressSpaceTypeRemapper &TypeMapper) {
  unsigned AddressSpace = Source->getAddressSpace();
  llvm::Type *MappedPointer = TypeMapper.remapType(Source->getType());
  AddressSpace = llvm::cast<llvm::PointerType>(MappedPointer)->getAddressSpace();
  auto *Mapped = new llvm::GlobalVariable(
      *Source->getParent(), TypeMapper.remapType(Source->getValueType()),
      Source->isConstant(), Source->getLinkage(), nullptr,
      Source->getName() + ".as-remapped", nullptr, Source->getThreadLocalMode(),
      AddressSpace, Source->isExternallyInitialized());
  Mapped->copyAttributesFrom(Source);
  Mapped->copyMetadata(Source, 0);
  return Mapped;
}

llvm::Function *createMappedFunction(
    llvm::Function *Source, AddressSpaceTypeRemapper &TypeMapper) {
  auto *FunctionType =
      llvm::cast<llvm::FunctionType>(TypeMapper.remapType(
          Source->getFunctionType()));
  auto *Mapped = llvm::Function::Create(
      FunctionType, Source->getLinkage(), Source->getAddressSpace(),
      Source->getName() + ".as-remapped", Source->getParent());
  Mapped->copyAttributesFrom(Source);
  Mapped->setCallingConv(Source->getCallingConv());
  return Mapped;
}

void mapGlobalInitializers(
    llvm::ArrayRef<GlobalPair> Globals, llvm::ValueToValueMapTy &GlobalMap,
    AddressSpaceTypeRemapper &TypeMapper) {
  for (const GlobalPair &Pair : Globals) {
    llvm::GlobalVariable *Source = Pair.first;
    if (!Source->hasInitializer())
      continue;
    llvm::Value *Mapped = llvm::MapValue(Source->getInitializer(), GlobalMap,
                                         llvm::RF_None, &TypeMapper);
    Pair.second->setInitializer(llvm::cast<llvm::Constant>(Mapped));
  }
}

void cloneFunctions(llvm::ArrayRef<FunctionPair> Functions,
                    llvm::ValueToValueMapTy &GlobalMap,
                    AddressSpaceTypeRemapper &TypeMapper) {
  for (const FunctionPair &Pair : Functions) {
    llvm::Function *Source = Pair.first;
    llvm::Function *Mapped = Pair.second;
    if (Source->isDeclaration())
      continue;

    llvm::ValueToValueMapTy LocalMap;
    for (const auto &Mapping : GlobalMap)
      LocalMap[Mapping.first] = Mapping.second;
    auto MappedArg = Mapped->arg_begin();
    for (llvm::Argument &SourceArg : Source->args())
      LocalMap[&SourceArg] = &*MappedArg++;
    llvm::SmallVector<llvm::ReturnInst *, 4> Returns;
    llvm::CloneFunctionInto(
        Mapped, Source, LocalMap,
        llvm::CloneFunctionChangeType::GlobalChanges, Returns, "", nullptr,
        &TypeMapper);
  }
}

void remapNamedMetadata(llvm::Module &Module,
                        llvm::ValueToValueMapTy &GlobalMap,
                        AddressSpaceTypeRemapper &TypeMapper) {
  for (llvm::NamedMDNode &Named : Module.named_metadata()) {
    for (unsigned Index = 0; Index < Named.getNumOperands(); ++Index) {
      llvm::MDNode *Mapped = llvm::MapMetadata(
          Named.getOperand(Index), GlobalMap, llvm::RF_IgnoreMissingLocals,
          &TypeMapper);
      Named.setOperand(Index, Mapped);
    }
  }
}

void replaceOriginals(llvm::ArrayRef<FunctionPair> Functions,
                      llvm::ArrayRef<GlobalPair> Globals) {
  for (const FunctionPair &Pair : Functions)
    Pair.first->dropAllReferences();
  for (const GlobalPair &Pair : Globals)
    Pair.first->setInitializer(nullptr);

  for (const FunctionPair &Pair : Functions) {
    Pair.second->takeName(Pair.first);
    Pair.first->eraseFromParent();
  }
  for (const GlobalPair &Pair : Globals) {
    Pair.second->takeName(Pair.first);
    Pair.first->eraseFromParent();
  }
}

} // namespace

bool pocl_llvm_remap_address_space(llvm::Module &Module, unsigned Source,
                                  unsigned Target, std::string &Error) {
  if (Source == Target)
    return false;
  if (!Module.alias_empty() || !Module.ifunc_empty()) {
    Error.append("SPIR-V address-space remapping does not support aliases or "
                 "indirect functions\n");
    return true;
  }

  AddressSpaceTypeRemapper TypeMapper(Source, Target);
  llvm::ValueToValueMapTy GlobalMap;
  llvm::SmallVector<GlobalPair, 16> Globals;
  llvm::SmallVector<FunctionPair, 32> Functions;
  llvm::SmallVector<llvm::GlobalVariable *, 16> OriginalGlobals;
  llvm::SmallVector<llvm::Function *, 32> OriginalFunctions;

  for (llvm::GlobalVariable &Global : Module.globals())
    OriginalGlobals.push_back(&Global);
  for (llvm::Function &Function : Module)
    OriginalFunctions.push_back(&Function);

  for (llvm::GlobalVariable *Global : OriginalGlobals) {
    auto *Mapped = createMappedGlobal(Global, TypeMapper);
    GlobalMap[Global] = Mapped;
    Globals.emplace_back(Global, Mapped);
  }
  for (llvm::Function *Function : OriginalFunctions) {
    auto *Mapped = createMappedFunction(Function, TypeMapper);
    GlobalMap[Function] = Mapped;
    Functions.emplace_back(Function, Mapped);
  }

  mapGlobalInitializers(Globals, GlobalMap, TypeMapper);
  cloneFunctions(Functions, GlobalMap, TypeMapper);
  remapNamedMetadata(Module, GlobalMap, TypeMapper);
  replaceOriginals(Functions, Globals);

  llvm::raw_string_ostream Stream(Error);
  return llvm::verifyModule(Module, &Stream);
}
