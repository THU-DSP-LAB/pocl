#include "pocl-ptx-gen-internal.h"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/IR/ReplaceConstant.h>

namespace pocl::cuda {
namespace {

using namespace llvm;

constexpr unsigned GenericSpace = 0;
constexpr unsigned SharedSpace = 3;
constexpr uint32_t NamedLocalLiveBit = uint32_t{1} << 31;
constexpr uint64_t GenericLocalTag = uint64_t{2} << 62;

bool isNamedLocal(Type *Ty) {
  return Ty->isPointerTy() && Ty->getPointerAddressSpace() == SharedSpace;
}

Value *namedLocal(IRBuilder<> &Builder, Value *Offset) {
  auto *Encoded = Builder.CreateAdd(Offset, Builder.getInt32(NamedLocalLiveBit));
  return Builder.CreateIntToPtr(Encoded,
                               PointerType::get(Builder.getContext(), SharedSpace));
}

Value *physicalOffset(IRBuilder<> &Builder, Value *Pointer) {
  auto *Bits = Builder.CreatePtrToInt(Pointer, Builder.getInt32Ty());
  return Builder.CreateSub(Bits, Builder.getInt32(NamedLocalLiveBit));
}

Value *genericToNamedLocal(IRBuilder<> &Builder, Value *Pointer) {
  auto *Offset = Builder.CreatePtrToInt(Pointer, Builder.getInt32Ty());
  auto *Logical = namedLocal(Builder, Offset);
  auto *Null = Builder.CreateIsNull(Pointer);
  return Builder.CreateSelect(Null, Constant::getNullValue(Logical->getType()), Logical);
}

Value *namedLocalToGeneric(IRBuilder<> &Builder, Value *Pointer) {
  auto *Offset = physicalOffset(Builder, Pointer);
  auto *Address = Builder.CreateOr(Builder.CreateZExt(Offset, Builder.getInt64Ty()),
                                   Builder.getInt64(GenericLocalTag));
  auto *Generic = Builder.CreateIntToPtr(
      Address, PointerType::get(Builder.getContext(), GenericSpace));
  auto *Null = Builder.CreateIsNull(Pointer);
  return Builder.CreateSelect(Null, Constant::getNullValue(Generic->getType()), Generic);
}

void normalizeCast(AddrSpaceCastInst &Cast) {
  IRBuilder<> Builder(&Cast);
  Value *Input = Cast.getPointerOperand();
  Value *Replacement = nullptr;
  if (Cast.getSrcAddressSpace() == GenericSpace && isNamedLocal(Cast.getType()))
    Replacement = genericToNamedLocal(Builder, Input);
  if (Cast.getDestAddressSpace() == GenericSpace && isNamedLocal(Input->getType()))
    Replacement = namedLocalToGeneric(Builder, Input);
  if (Replacement) {
    Cast.replaceAllUsesWith(Replacement);
    Cast.eraseFromParent();
  }
}

void encodeStaticBase(Instruction &Instruction) {
  for (auto &Use : Instruction.operands()) {
    auto *Global = dyn_cast<GlobalVariable>(Use.get());
    if (!Global || !isNamedLocal(Global->getType())) continue;
    auto *Insertion = &Instruction;
    if (auto *Phi = dyn_cast<PHINode>(&Instruction))
      Insertion = Phi->getIncomingBlock(Use)->getTerminator();
    IRBuilder<NoFolder> Builder(Insertion);
    auto *Offset = Builder.CreatePtrToInt(Global, Builder.getInt32Ty());
    auto *Encoded = Builder.CreateAdd(Offset, Builder.getInt32(NamedLocalLiveBit));
    auto *Result = Builder.CreateIntToPtr(Encoded, Global->getType());
    Use.set(Result);
  }
}

void decodeMemoryPointer(Instruction &Instruction, unsigned Index) {
  auto *Pointer = Instruction.getOperand(Index);
  if (!isNamedLocal(Pointer->getType())) return;
  IRBuilder<> Builder(&Instruction);
  auto *Offset = physicalOffset(Builder, Pointer);
  Instruction.setOperand(Index, Builder.CreateIntToPtr(Offset, Pointer->getType()));
}

void lowerMemoryAccess(Instruction &Instruction) {
  if (isa<LoadInst, AtomicRMWInst, AtomicCmpXchgInst>(&Instruction))
    return decodeMemoryPointer(Instruction, 0);
  if (isa<StoreInst>(&Instruction))
    return decodeMemoryPointer(Instruction, 1);
  if (isa<MemTransferInst>(&Instruction))
    decodeMemoryPointer(Instruction, 1);
  if (isa<MemIntrinsic>(&Instruction))
    decodeMemoryPointer(Instruction, 0);
}

} // namespace

void lowerGpidlNamedLocalPointers(llvm::Module &Module) {
  llvm::SmallVector<llvm::Constant *, 16> SharedGlobals;
  for (auto &Global : Module.globals())
    if (isNamedLocal(Global.getType()))
      SharedGlobals.push_back(&Global);
  llvm::convertUsersOfConstantsToInstructions(SharedGlobals);

  llvm::SmallVector<llvm::Instruction *, 64> Instructions;
  for (auto &Function : Module)
    for (auto &Block : Function)
      for (auto &Instruction : Block)
        Instructions.push_back(&Instruction);

  for (auto *Instruction : Instructions) {
    encodeStaticBase(*Instruction);
    if (auto *Address = llvm::dyn_cast<llvm::GetElementPtrInst>(Instruction);
        Address && isNamedLocal(Address->getType()))
      Address->setIsInBounds(false);
    if (auto *Cast = llvm::dyn_cast<llvm::AddrSpaceCastInst>(Instruction))
      normalizeCast(*Cast);
    else
      lowerMemoryAccess(*Instruction);
  }
}

} // namespace pocl::cuda
