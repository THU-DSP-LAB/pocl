/* pocl-ptx-rotate.cc: NVPTX 64-bit rotate lowering.

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl-ptx-rotate.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#include <vector>

static constexpr llvm::StringLiteral RotatePrefix("_Z10_cl_rotate");
static constexpr llvm::StringLiteral RotateAssembly(R"({
  .reg .u32 %amount;
  .reg .u32 %inverse;
  .reg .b64 %left;
  .reg .b64 %right;
  cvt.u32.u64 %amount, $2;
  and.b32 %amount, %amount, 63;
  shl.b64 %left, $1, %amount;
  sub.u32 %inverse, 64, %amount;
  shr.u64 %right, $1, %inverse;
  or.b64 $0, %left, %right;
})");

static bool isI64Shape(const llvm::Type *Type) {
  if (Type->isIntegerTy(64))
    return true;
  const auto *Vector = llvm::dyn_cast<llvm::FixedVectorType>(Type);
  return Vector && Vector->getElementType()->isIntegerTy(64);
}

static bool isI64Rotate(const llvm::Function &Function) {
  if (!Function.getName().starts_with(RotatePrefix) ||
      Function.isDeclaration() || Function.arg_size() != 2)
    return false;
  const llvm::Type *ReturnType = Function.getReturnType();
  return isI64Shape(ReturnType) &&
         Function.getArg(0)->getType() == ReturnType &&
         Function.getArg(1)->getType() == ReturnType;
}

static llvm::InlineAsm *createRotateAssembly(llvm::LLVMContext &Context) {
  llvm::Type *I64 = llvm::Type::getInt64Ty(Context);
  llvm::FunctionType *Type = llvm::FunctionType::get(I64, {I64, I64}, false);
  return llvm::InlineAsm::get(Type, RotateAssembly, "=l,l,l", false);
}

struct RotateMatch {
  llvm::Value *Value = nullptr;
  llvm::Value *Amount = nullptr;

  explicit operator bool() const { return Value != nullptr; }
};

static llvm::Value *getMaskedAmount(llvm::Value *Value) {
  auto *Mask = llvm::dyn_cast<llvm::BinaryOperator>(Value);
  if (!Mask || Mask->getOpcode() != llvm::Instruction::And)
    return nullptr;
  for (unsigned Operand = 0; Operand < 2; ++Operand) {
    const auto *Constant =
        llvm::dyn_cast<llvm::ConstantInt>(Mask->getOperand(Operand));
    if (Constant && Constant->equalsInt(63))
      return Mask->getOperand(1 - Operand);
  }
  return nullptr;
}

static bool isInverseAmount(llvm::Value *RightAmount, llvm::Value *Amount,
                            llvm::Value *LeftAmount) {
  llvm::Value *RightBase = getMaskedAmount(RightAmount);
  if (RightBase) {
    auto *Negate = llvm::dyn_cast<llvm::BinaryOperator>(RightBase);
    const auto *Zero =
        Negate ? llvm::dyn_cast<llvm::ConstantInt>(Negate->getOperand(0))
               : nullptr;
    return Negate && Negate->getOpcode() == llvm::Instruction::Sub && Zero &&
           Zero->isZero() && Negate->getOperand(1) == Amount;
  }

  auto *Subtract = llvm::dyn_cast<llvm::BinaryOperator>(RightAmount);
  const auto *Width =
      Subtract ? llvm::dyn_cast<llvm::ConstantInt>(Subtract->getOperand(0))
               : nullptr;
  return Subtract && Subtract->getOpcode() == llvm::Instruction::Sub && Width &&
         Width->equalsInt(64) && Subtract->getOperand(1) == LeftAmount;
}

static RotateMatch matchShiftPair(llvm::Value *LeftValue,
                                  llvm::Value *RightValue) {
  auto *Left = llvm::dyn_cast<llvm::BinaryOperator>(LeftValue);
  auto *Right = llvm::dyn_cast<llvm::BinaryOperator>(RightValue);
  if (!Left || !Right || Left->getOpcode() != llvm::Instruction::Shl ||
      Right->getOpcode() != llvm::Instruction::LShr ||
      Left->getOperand(0) != Right->getOperand(0))
    return {};

  llvm::Value *Amount = getMaskedAmount(Left->getOperand(1));
  if (!Amount ||
      !isInverseAmount(Right->getOperand(1), Amount, Left->getOperand(1)))
    return {};
  return {Left->getOperand(0), Amount};
}

static RotateMatch matchRotateExpression(llvm::BinaryOperator &Combine) {
  if (!Combine.getType()->isIntegerTy(64) ||
      (Combine.getOpcode() != llvm::Instruction::Or &&
       Combine.getOpcode() != llvm::Instruction::Add))
    return {};
  RotateMatch Match =
      matchShiftPair(Combine.getOperand(0), Combine.getOperand(1));
  return Match ? Match
               : matchShiftPair(Combine.getOperand(1), Combine.getOperand(0));
}

static llvm::Value *lowerVectorRotate(llvm::Function &Function,
                                      llvm::InlineAsm &Assembly,
                                      llvm::IRBuilder<> &Builder) {
  auto *VectorType =
      llvm::cast<llvm::FixedVectorType>(Function.getReturnType());
  llvm::Value *Result = llvm::PoisonValue::get(VectorType);
  for (unsigned Index = 0; Index < VectorType->getNumElements(); ++Index) {
    llvm::Value *Value =
        Builder.CreateExtractElement(Function.getArg(0), Index);
    llvm::Value *Amount =
        Builder.CreateExtractElement(Function.getArg(1), Index);
    llvm::Value *Rotated = Builder.CreateCall(&Assembly, {Value, Amount});
    Result = Builder.CreateInsertElement(Result, Rotated, Index);
  }
  return Result;
}

static void lowerRotate(llvm::Function &Function, llvm::InlineAsm &Assembly) {
  Function.deleteBody();
  llvm::BasicBlock *Entry =
      llvm::BasicBlock::Create(Function.getContext(), "entry", &Function);
  llvm::IRBuilder<> Builder(Entry);
  llvm::Value *Result =
      Function.getReturnType()->isIntegerTy(64)
          ? Builder.CreateCall(&Assembly,
                               {Function.getArg(0), Function.getArg(1)})
          : lowerVectorRotate(Function, Assembly, Builder);
  Builder.CreateRet(Result);
}

static unsigned lowerRotateExpressions(llvm::Module &Module,
                                       llvm::InlineAsm &Assembly) {
  std::vector<llvm::BinaryOperator *> Work;
  for (llvm::Function &Function : Module)
    for (llvm::BasicBlock &Block : Function)
      for (llvm::Instruction &Instruction : Block)
        if (auto *Combine = llvm::dyn_cast<llvm::BinaryOperator>(&Instruction))
          if (matchRotateExpression(*Combine))
            Work.push_back(Combine);

  for (llvm::BinaryOperator *Combine : Work) {
    RotateMatch Match = matchRotateExpression(*Combine);
    llvm::IRBuilder<> Builder(Combine);
    llvm::Value *Replacement =
        Builder.CreateCall(&Assembly, {Match.Value, Match.Amount});
    Combine->replaceAllUsesWith(Replacement);
    Combine->eraseFromParent();
  }
  return Work.size();
}

static bool isI64RotateIntrinsic(const llvm::CallInst &Call) {
  const llvm::Function *Callee = Call.getCalledFunction();
  return Callee && Callee->getIntrinsicID() == llvm::Intrinsic::fshl &&
         Call.getType()->isIntegerTy(64) &&
         Call.getArgOperand(0) == Call.getArgOperand(1);
}

static unsigned lowerRotateIntrinsics(llvm::Module &Module,
                                      llvm::InlineAsm &Assembly) {
  std::vector<llvm::CallInst *> Work;
  for (llvm::Function &Function : Module)
    for (llvm::BasicBlock &Block : Function)
      for (llvm::Instruction &Instruction : Block)
        if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&Instruction))
          if (isI64RotateIntrinsic(*Call))
            Work.push_back(Call);

  for (llvm::CallInst *Call : Work) {
    llvm::IRBuilder<> Builder(Call);
    llvm::Value *Replacement = Builder.CreateCall(
        &Assembly, {Call->getArgOperand(0), Call->getArgOperand(2)});
    Call->replaceAllUsesWith(Replacement);
    Call->eraseFromParent();
  }
  return Work.size();
}

unsigned pocl_cuda_lower_i64_rotates(llvm::Module *Module) {
  llvm::InlineAsm *Assembly = createRotateAssembly(Module->getContext());
  unsigned Lowered = 0;
  for (llvm::Function &Function : *Module) {
    if (!isI64Rotate(Function))
      continue;
    lowerRotate(Function, *Assembly);
    ++Lowered;
  }
  return Lowered + lowerRotateExpressions(*Module, *Assembly) +
         lowerRotateIntrinsics(*Module, *Assembly);
}
