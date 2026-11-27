#include "pocl-ptx-gen-internal.h"

#include <array>
#include <optional>
#include <vector>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

namespace pocl::cuda {
namespace {

constexpr unsigned Generic = 0;
constexpr unsigned Global = 1;
constexpr unsigned Shared = 3;
constexpr unsigned ThreadPrivate = 5;
constexpr uint64_t PrivateBit = uint64_t{1} << 62;

struct MemoryAccess {
  llvm::Instruction *Operation;
  unsigned PointerIndex;
  bool SupportsPrivate;
};

std::optional<MemoryAccess> describe(llvm::Instruction &Instruction) {
  if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(&Instruction))
    return MemoryAccess{Load, Load->getPointerOperandIndex(), !Load->isAtomic()};
  if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&Instruction))
    return MemoryAccess{Store, Store->getPointerOperandIndex(), !Store->isAtomic()};
  if (auto *Rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&Instruction))
    return MemoryAccess{Rmw, Rmw->getPointerOperandIndex(), false};
  if (auto *Exchange = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&Instruction))
    return MemoryAccess{Exchange, Exchange->getPointerOperandIndex(), false};
  return std::nullopt;
}

bool isGenericAddress(const MemoryAccess &Access) {
  return Access.Operation->getOperand(Access.PointerIndex)->getType()
             ->getPointerAddressSpace() == Generic;
}

bool useKnownNamedAddress(const MemoryAccess &Access) {
  auto *Pointer = Access.Operation->getOperand(Access.PointerIndex);
  const auto *Base = llvm::getUnderlyingObject(Pointer);
  const unsigned AddressSpace = Base->getType()->getPointerAddressSpace();
  if (llvm::isa<llvm::AllocaInst>(Base)) return true;
  if (AddressSpace != Global && AddressSpace != Shared && AddressSpace != ThreadPrivate)
    return false;
  llvm::IRBuilder<> Builder(Access.Operation);
  auto *Named = Builder.CreateAddrSpaceCast(
      Pointer, llvm::PointerType::get(Builder.getContext(), AddressSpace));
  Access.Operation->setOperand(Access.PointerIndex, Named);
  return true;
}

struct Space {
  unsigned Tag;
  unsigned AddressSpace;
  const char *Name;
};

llvm::Value *namedPointer(llvm::IRBuilder<> &Builder, llvm::Value *Address,
                          unsigned AddressSpace) {
  if (AddressSpace != Global)
    Address = Builder.CreateTrunc(Address, Builder.getInt32Ty());
  return Builder.CreateIntToPtr(
      Address, llvm::PointerType::get(Builder.getContext(), AddressSpace));
}

llvm::Instruction *appendAccess(const MemoryAccess &Access, llvm::Value *Pointer,
                                llvm::BasicBlock &Block) {
  auto *Copy = Access.Operation->clone();
  Copy->setOperand(Access.PointerIndex, Pointer);
  Copy->insertInto(&Block, Block.end());
  return Copy;
}

void partition(const MemoryAccess &Access) {
  auto &Operation = *Access.Operation;
  auto &Entry = *Operation.getParent();
  auto &Context = Operation.getContext();
  auto *Continuation = Entry.splitBasicBlock(Operation.getIterator(), "flat.cont");
  Entry.getTerminator()->eraseFromParent();

  llvm::IRBuilder<> Dispatch(&Entry);
  auto *OriginalPointer = Operation.getOperand(Access.PointerIndex);
  auto *Address = Dispatch.CreatePtrToInt(Operation.getOperand(Access.PointerIndex),
                                           Dispatch.getInt64Ty(), "flat.address");
  auto *Invalid = llvm::BasicBlock::Create(Context, "flat.invalid", Entry.getParent(), Continuation);
  std::array<llvm::BasicBlock *, 4> Destinations{Invalid, Invalid, Invalid, Invalid};
  llvm::PHINode *Result = Operation.getType()->isVoidTy() ? nullptr
      : llvm::PHINode::Create(Operation.getType(), Access.SupportsPrivate ? 4 : 3, "flat.result",
                              &*Continuation->begin());

  constexpr std::array Spaces = {
      Space{0, Global, "flat.global"}, Space{2, Shared, "flat.shared"},
      Space{1, ThreadPrivate, "flat.private"}};
  for (const auto &Space : Spaces) {
    if (!Access.SupportsPrivate && Space.AddressSpace == ThreadPrivate) continue;
    auto *Block = llvm::BasicBlock::Create(Context, Space.Name, Entry.getParent(), Continuation);
    Destinations[Space.Tag] = Block;
    llvm::IRBuilder<> Builder(Block);
    auto *Named = namedPointer(Builder, Address, Space.AddressSpace);
    auto *ResultHere = appendAccess(Access, Named, *Block);
    Builder.SetInsertPoint(Block);
    Builder.CreateBr(Continuation);
    if (Result) Result->addIncoming(ResultHere, Block);
  }
  auto *LocalBit = Dispatch.CreateICmpSLT(Address, Dispatch.getInt64(0));
  if (!Access.SupportsPrivate) {
    // OpenCL atomics operate only on global or local memory, even through a
    // generic pointer. The sign bit distinguishes these two valid spaces.
    Dispatch.CreateCondBr(LocalBit, Destinations[2], Destinations[0]);
    Invalid->eraseFromParent();
    if (Result) Operation.replaceAllUsesWith(Result);
    Operation.eraseFromParent();
    return;
  }

  // Test each tag bit directly. A multiway switch here can become an
  // indirect uniform branch in SASS even when the PTX uses direct branches.
  auto *Low = llvm::BasicBlock::Create(Context, "flat.low", Entry.getParent(), Continuation);
  auto *High = llvm::BasicBlock::Create(Context, "flat.high", Entry.getParent(), Continuation);
  auto *OddBit = Dispatch.CreateIsNotNull(
      Dispatch.CreateAnd(Address, Dispatch.getInt64(PrivateBit)));
  Dispatch.CreateCondBr(LocalBit, High, Low);
  llvm::IRBuilder<>(Low).CreateCondBr(OddBit, Destinations[1], Destinations[0]);
  llvm::IRBuilder<>(High).CreateCondBr(OddBit, Destinations[3], Destinations[2]);

  // Keep the existing diagnostic for an invalid tag. It must not reach a
  // named memory operation through a guessed address space.
  auto *InvalidAccess = appendAccess(Access, OriginalPointer, *Invalid);
  llvm::IRBuilder<>(Invalid).CreateBr(Continuation);
  if (Result) {
    Result->addIncoming(InvalidAccess, Invalid);
    Operation.replaceAllUsesWith(Result);
  }
  Operation.eraseFromParent();
}

} // namespace

void partitionGpidlFlatMemory(llvm::Module &Module) {
  std::vector<MemoryAccess> Accesses;
  for (auto &Function : Module)
    for (auto &Block : Function)
      for (auto &Instruction : Block)
        if (auto Access = describe(Instruction); Access && isGenericAddress(*Access))
          Accesses.push_back(*Access);
  for (const auto &Access : Accesses)
    if (!useKnownNamedAddress(Access)) partition(Access);
}

} // namespace pocl::cuda
