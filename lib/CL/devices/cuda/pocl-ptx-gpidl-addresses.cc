#include "pocl-ptx-gen-internal.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace pocl::cuda {
namespace {

llvm::Constant *normalizeNullCast(const llvm::ConstantExpr &Expression,
                                 llvm::ArrayRef<llvm::Constant *> Operands) {
  if (Expression.getOpcode() == llvm::Instruction::AddrSpaceCast &&
      llvm::isa<llvm::ConstantPointerNull>(Operands.front()))
    return llvm::ConstantPointerNull::get(
        llvm::cast<llvm::PointerType>(Expression.getType()));
  return Expression.getWithOperands(Operands, Expression.getType());
}

llvm::Constant *normalizeConstant(llvm::Constant *Value) {
  if (llvm::isa<llvm::GlobalValue, llvm::BlockAddress>(Value) ||
      Value->getNumOperands() == 0)
    return Value;

  llvm::SmallVector<llvm::Constant *, 8> Operands;
  for (auto &Operand : Value->operands())
    Operands.push_back(normalizeConstant(llvm::cast<llvm::Constant>(Operand)));

  if (auto *Expression = llvm::dyn_cast<llvm::ConstantExpr>(Value))
    return normalizeNullCast(*Expression, Operands);
  if (auto *Record = llvm::dyn_cast<llvm::ConstantStruct>(Value))
    return llvm::ConstantStruct::get(Record->getType(), Operands);
  if (auto *Array = llvm::dyn_cast<llvm::ConstantArray>(Value))
    return llvm::ConstantArray::get(Array->getType(), Operands);
  if (llvm::isa<llvm::ConstantVector>(Value))
    return llvm::ConstantVector::get(Operands);
  return Value;
}

} // namespace

void normalizeGpidlNullCasts(llvm::Module &Module) {
  for (auto &Global : Module.globals())
    if (Global.hasInitializer())
      Global.setInitializer(normalizeConstant(Global.getInitializer()));

  for (auto &Function : Module)
    for (auto &Block : Function)
      for (auto &Instruction : Block)
        for (auto &Use : Instruction.operands())
          if (auto *Constant = llvm::dyn_cast<llvm::Constant>(Use.get()))
            Use.set(normalizeConstant(Constant));
}

} // namespace pocl::cuda
