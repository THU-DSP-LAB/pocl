/* pocl-ptx-vector-args.cc - CUDA kernel vector argument lowering

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
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

#include "pocl-ptx-vector-args.h"

#include "LLVMUtils.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <vector>

struct VectorArgInfo {
  unsigned Index;
  llvm::Type *Type;
  llvm::Align Alignment;
};

// LLVM 18's NVPTX selector crashes while lowering kernels with multiple
// differently-sized vector parameters. PTX represents aggregate parameters as
// byte arrays, so pass vectors by value through an explicitly aligned pointer
// and load the original value in the kernel entry block.
static llvm::Function *
convertVectorArgsToByVal(llvm::Module *Module, llvm::Function *Function) {
  std::vector<llvm::Type *> ArgumentTypes;
  bool HasVectorArg = false;
  for (const auto &Arg : Function->args()) {
    llvm::Type *Type = Arg.getType();
    if (Type->isVectorTy()) {
      Type = llvm::PointerType::get(Module->getContext(), 0);
      HasVectorArg = true;
    }
    ArgumentTypes.push_back(Type);
  }
  if (!HasVectorArg)
    return nullptr;

  auto *NewType = llvm::FunctionType::get(
      Function->getReturnType(), ArgumentTypes, Function->isVarArg());
  auto *NewFunction = llvm::Function::Create(
      NewType, Function->getLinkage(), Function->getName(), Module);
  NewFunction->takeName(Function);

  llvm::ValueToValueMapTy ValueMap;
  llvm::SmallVector<llvm::LoadInst *, 8> VectorLoads;
  llvm::SmallVector<VectorArgInfo, 8> VectorArgs;
  auto NewArg = NewFunction->arg_begin();
  unsigned ArgIndex = 0;
  for (auto &OldArg : Function->args()) {
    NewArg->setName(OldArg.getName());
    llvm::Type *OldType = OldArg.getType();
    if (!OldType->isVectorTy()) {
      ValueMap[&OldArg] = &*NewArg;
    } else {
      llvm::Align Alignment
          = Module->getDataLayout().getABITypeAlign(OldType);
      auto *Load = new llvm::LoadInst(OldType, &*NewArg,
                                      OldArg.getName() + ".value", false,
                                      Alignment);
      ValueMap[&OldArg] = Load;
      VectorLoads.push_back(Load);
      VectorArgs.push_back({ArgIndex, OldType, Alignment});
    }
    ++NewArg;
    ++ArgIndex;
  }

  llvm::SmallVector<llvm::ReturnInst *, 1> Returns;
  CloneFunctionIntoAbs(NewFunction, Function, ValueMap, Returns);
  for (const auto &Arg : VectorArgs) {
    NewFunction->addParamAttr(
        Arg.Index, llvm::Attribute::getWithByValType(Module->getContext(),
                                                     Arg.Type));
    NewFunction->addParamAttr(
        Arg.Index, llvm::Attribute::getWithAlignment(Module->getContext(),
                                                     Arg.Alignment));
  }
  llvm::Instruction *InsertionPoint
      = &*NewFunction->getEntryBlock().getFirstInsertionPt();
  for (auto *Load : VectorLoads)
    Load->insertBefore(InsertionPoint);
  return NewFunction;
}

void pocl_cuda_fix_vector_args(llvm::Module *Module) {
  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;
  for (auto &Function : Module->functions()) {
    if (!pocl::isKernelToProcess(Function))
      continue;
    if (convertVectorArgsToByVal(Module, &Function))
      FunctionsToErase.push_back(&Function);
  }
  for (auto *Function : FunctionsToErase)
    Function->eraseFromParent();
}
