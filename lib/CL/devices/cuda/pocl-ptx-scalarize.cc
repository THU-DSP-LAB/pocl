/* pocl-ptx-scalarize.cc: explicit vector scalarization for NVPTX.

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

#include "pocl-ptx-scalarize.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"

#include <utility>
#include <vector>

static llvm::Value *createByteAddress(llvm::IRBuilder<> &Builder,
                                      llvm::Value *Base, unsigned Index) {
  if (Index == 0)
    return Base;
  return Builder.CreateConstGEP1_32(Builder.getInt8Ty(), Base, Index);
}

static llvm::Value *scalarizeI8Load(llvm::LoadInst *Load,
                                    llvm::FixedVectorType *VectorType) {
  llvm::IRBuilder<> Builder(Load);
  llvm::Value *Result = llvm::PoisonValue::get(VectorType);
  for (unsigned Index = 0; Index < VectorType->getNumElements(); ++Index) {
    llvm::Value *Address =
        createByteAddress(Builder, Load->getPointerOperand(), Index);
    llvm::Align Alignment = llvm::commonAlignment(Load->getAlign(), Index);
    // LLVM 18's NVPTX load/store vectorizer reverses lanes for adjacent i8
    // operations. Volatile keeps these correctness-critical byte accesses
    // scalar through SelectionDAG lowering.
    llvm::LoadInst *Element = Builder.CreateAlignedLoad(
        Builder.getInt8Ty(), Address, Alignment, true);
    Element->copyMetadata(*Load);
    Result = Builder.CreateInsertElement(Result, Element, Index);
  }
  return Result;
}

static void scalarizeI8Store(llvm::StoreInst *Store,
                             llvm::FixedVectorType *VectorType) {
  llvm::IRBuilder<> Builder(Store);
  llvm::Value *Vector = Store->getValueOperand();
  for (unsigned Index = 0; Index < VectorType->getNumElements(); ++Index) {
    llvm::Value *Address =
        createByteAddress(Builder, Store->getPointerOperand(), Index);
    llvm::Value *Element = Builder.CreateExtractElement(Vector, Index);
    llvm::Align Alignment = llvm::commonAlignment(Store->getAlign(), Index);
    llvm::StoreInst *Scalar =
        Builder.CreateAlignedStore(Element, Address, Alignment, true);
    Scalar->copyMetadata(*Store);
  }
}

static bool isI8Vector(const llvm::Type *Type) {
  const auto *VectorType = llvm::dyn_cast<llvm::FixedVectorType>(Type);
  return VectorType && VectorType->getElementType()->isIntegerTy(8);
}

static void scalarizeI8VectorMemory(llvm::Module &Module) {
  std::vector<llvm::Instruction *> Work;
  for (llvm::Function &Function : Module)
    for (llvm::BasicBlock &Block : Function)
      for (llvm::Instruction &Instruction : Block)
        if ((llvm::isa<llvm::LoadInst>(Instruction) &&
             isI8Vector(Instruction.getType())) ||
            (llvm::isa<llvm::StoreInst>(Instruction) &&
             isI8Vector(Instruction.getOperand(0)->getType())))
          Work.push_back(&Instruction);

  for (llvm::Instruction *Instruction : Work) {
    if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(Instruction)) {
      auto *VectorType = llvm::cast<llvm::FixedVectorType>(Load->getType());
      Load->replaceAllUsesWith(scalarizeI8Load(Load, VectorType));
    } else {
      auto *Store = llvm::cast<llvm::StoreInst>(Instruction);
      auto *VectorType = llvm::cast<llvm::FixedVectorType>(
          Store->getValueOperand()->getType());
      scalarizeI8Store(Store, VectorType);
    }
    Instruction->eraseFromParent();
  }
}

static void widenI8Comparisons(llvm::Module &Module) {
  std::vector<llvm::ICmpInst *> Work;
  for (llvm::Function &Function : Module)
    for (llvm::BasicBlock &Block : Function)
      for (llvm::Instruction &Instruction : Block)
        if (auto *Compare = llvm::dyn_cast<llvm::ICmpInst>(&Instruction))
          if (Compare->getOperand(0)->getType()->isIntegerTy(8))
            Work.push_back(Compare);

  for (llvm::ICmpInst *Compare : Work) {
    llvm::IRBuilder<> Builder(Compare);
    llvm::Value *Left =
        Compare->isSigned()
            ? Builder.CreateSExt(Compare->getOperand(0), Builder.getInt16Ty())
            : Builder.CreateZExt(Compare->getOperand(0), Builder.getInt16Ty());
    llvm::Value *Right =
        Compare->isSigned()
            ? Builder.CreateSExt(Compare->getOperand(1), Builder.getInt16Ty())
            : Builder.CreateZExt(Compare->getOperand(1), Builder.getInt16Ty());
    llvm::Value *Replacement =
        Builder.CreateICmp(Compare->getPredicate(), Left, Right);
    Compare->replaceAllUsesWith(Replacement);
    Compare->eraseFromParent();
  }
}

void pocl_cuda_scalarize_vectors(llvm::Module *Module) {
  scalarizeI8VectorMemory(*Module);
  llvm::LoopAnalysisManager LoopAnalyses;
  llvm::FunctionAnalysisManager FunctionAnalyses;
  llvm::CGSCCAnalysisManager CGSCCAnalyses;
  llvm::ModuleAnalysisManager ModuleAnalyses;
  llvm::PassBuilder Builder;
  Builder.registerModuleAnalyses(ModuleAnalyses);
  Builder.registerCGSCCAnalyses(CGSCCAnalyses);
  Builder.registerFunctionAnalyses(FunctionAnalyses);
  Builder.registerLoopAnalyses(LoopAnalyses);
  Builder.crossRegisterProxies(LoopAnalyses, FunctionAnalyses, CGSCCAnalyses,
                               ModuleAnalyses);

  llvm::ScalarizerPassOptions Options;
  Options.ScalarizeVariableInsertExtract = true;
  Options.ScalarizeLoadStore = true;
  llvm::FunctionPassManager Functions;
  Functions.addPass(llvm::ScalarizerPass(Options));
  llvm::ModulePassManager Modules;
  Modules.addPass(
      llvm::createModuleToFunctionPassAdaptor(std::move(Functions)));
  Modules.run(*Module, ModuleAnalyses);
  widenI8Comparisons(*Module);
}
