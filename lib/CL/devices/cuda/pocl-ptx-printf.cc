/* pocl-ptx-printf.cc: CUDA printf ABI lowering.

   Copyright (c) 2016 James Price / University of Bristol
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

#include "pocl-ptx-printf.h"

#include "LLVMUtils.h"
#include "pocl_debug.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"

#include <algorithm>
#include <vector>

namespace {

constexpr uint64_t PrintfArgumentBytes = 128;
constexpr unsigned PrintfArgumentAlignment = 128;

struct ArgumentStore {
  llvm::Value *Buffer;
  llvm::Value *Value;
  unsigned Index;
};

struct CallLoweringContext {
  llvm::Function *Printf;
  llvm::Type *FormatType;
  llvm::AllocaInst *ArgumentBuffer;
};

using PrintfCallsByFunction =
    llvm::MapVector<llvm::Function *, std::vector<llvm::CallInst *>>;

void eraseVarArgIntrinsics(llvm::Module *Module) {
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_start"));
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_end"));
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_start.p0"));
  pocl::eraseFunctionAndCallers(Module->getFunction("llvm.va_end.p0"));
}

llvm::AllocaInst *createArgumentIndex(llvm::Function *Printf) {
  llvm::IRBuilder<> Builder(&*Printf->getEntryBlock().begin());
  auto *Index = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr,
                                     "pocl.cuda.printf.arg.index");
  Index->setAlignment(llvm::Align(4));
  Builder.CreateAlignedStore(Builder.getInt32(0), Index, llvm::Align(4));
  return Index;
}

void replaceVaArgCalls(llvm::Module *Module, llvm::Function *Printf,
                       llvm::AllocaInst *ArgumentIndex) {
  llvm::Function *VaArg = Module->getFunction("__cl_va_arg");
  if (!VaArg)
    return;

  llvm::Argument *Arguments = std::next(Printf->arg_begin());
  std::vector<llvm::User *> Calls(VaArg->user_begin(), VaArg->user_end());
  for (llvm::User *User : Calls) {
    auto *Call = llvm::dyn_cast<llvm::CallInst>(User);
    if (!Call)
      continue;
    llvm::IRBuilder<> Builder(Call);
    llvm::Value *Index = Builder.CreateAlignedLoad(
        Builder.getInt32Ty(), ArgumentIndex, llvm::Align(4));
    llvm::Value *Offset =
        Builder.CreateMul(Index, Builder.getInt32(PrintfArgumentBytes));
    llvm::Value *Source =
        Builder.CreateInBoundsGEP(Builder.getInt8Ty(), Arguments, Offset);
    Builder.CreateMemCpy(Call->getArgOperand(1), llvm::Align(1), Source,
                         llvm::Align(PrintfArgumentAlignment),
                         PrintfArgumentBytes);
    llvm::Value *Next = Builder.CreateAdd(Index, Builder.getInt32(1));
    Builder.CreateAlignedStore(Next, ArgumentIndex, llvm::Align(4));
    Call->eraseFromParent();
  }
  VaArg->eraseFromParent();
}

llvm::Value *castPointerArgument(llvm::IRBuilder<> &Builder,
                                 llvm::Value *Argument) {
  auto *Type = llvm::dyn_cast<llvm::PointerType>(Argument->getType());
  if (!Type || Type->getAddressSpace() == 0)
    return Argument;
  return Builder.CreateAddrSpaceCast(
      Argument, llvm::PointerType::get(Builder.getContext(), 0));
}

llvm::Value *promoteScalarArgument(llvm::IRBuilder<> &Builder,
                                   llvm::Value *Argument) {
  llvm::Type *Type = Argument->getType();
  if (!Type->isFloatingPointTy() || Type->isDoubleTy())
    return Argument;
  return Builder.CreateFPExt(Argument, Builder.getDoubleTy(),
                             "pocl.cuda.printf.fp.promoted");
}

bool storeArgument(llvm::IRBuilder<> &Builder, const ArgumentStore &Store) {
  llvm::Value *Argument = promoteScalarArgument(
      Builder, castPointerArgument(Builder, Store.Value));
  llvm::Type *Type = Argument->getType();
  llvm::TypeSize Size =
      Builder.GetInsertBlock()->getModule()->getDataLayout().getTypeStoreSize(
          Type);
  if (Size.isScalable() || Size.getFixedValue() > PrintfArgumentBytes) {
    POCL_MSG_ERR("[CUDA] printf argument %u occupies %llu bytes; maximum is "
                 "%llu\n",
                 Store.Index, (unsigned long long)Size.getKnownMinValue(),
                 (unsigned long long)PrintfArgumentBytes);
    return false;
  }

  llvm::Value *Offset = Builder.getInt64(static_cast<uint64_t>(Store.Index) *
                                         PrintfArgumentBytes);
  llvm::Value *Slot =
      Builder.CreateInBoundsGEP(Builder.getInt8Ty(), Store.Buffer, Offset);
  llvm::Align Alignment =
      Builder.GetInsertBlock()->getModule()->getDataLayout().getABITypeAlign(
          Type);
  Builder.CreateAlignedStore(Argument, Slot, Alignment);
  return true;
}

bool replaceCall(llvm::CallInst *Call, const CallLoweringContext &Context) {
  llvm::IRBuilder<> Builder(Call);
  unsigned ArgumentCount = Call->arg_size() - 1;

  for (unsigned Index = 0; Index < ArgumentCount; ++Index)
    if (!storeArgument(Builder, {Context.ArgumentBuffer,
                                 Call->getArgOperand(Index + 1), Index}))
      return false;

  llvm::Value *Format = Call->getArgOperand(0);
  if (llvm::isa<llvm::UndefValue>(Format))
    Format = llvm::UndefValue::get(Context.FormatType);
  llvm::CallInst *Replacement =
      Builder.CreateCall(Context.Printf, {Format, Context.ArgumentBuffer});
  Replacement->setCallingConv(Call->getCallingConv());
  Replacement->setDebugLoc(Call->getDebugLoc());
  Call->replaceAllUsesWith(Replacement);
  Call->eraseFromParent();
  return true;
}

PrintfCallsByFunction collectPrintfCalls(llvm::Function *Printf) {
  PrintfCallsByFunction CallsByFunction;
  std::vector<llvm::User *> Users(Printf->user_begin(), Printf->user_end());
  for (llvm::User *User : Users)
    if (auto *Call = llvm::dyn_cast<llvm::CallInst>(User))
      CallsByFunction[Call->getFunction()].push_back(Call);
  return CallsByFunction;
}

llvm::AllocaInst *
createArgumentBuffer(llvm::Function *Function,
                     const std::vector<llvm::CallInst *> &Calls) {
  uint64_t SlotCount = 1;
  for (const llvm::CallInst *Call : Calls)
    SlotCount = std::max<uint64_t>(SlotCount, Call->arg_size() - 1);

  llvm::IRBuilder<> Builder(&*Function->getEntryBlock().getFirstInsertionPt());
  auto *Buffer = Builder.CreateAlloca(
      Builder.getInt8Ty(), Builder.getInt64(SlotCount * PrintfArgumentBytes),
      "pocl.cuda.printf.args");
  Buffer->setAlignment(llvm::Align(PrintfArgumentAlignment));
  return Buffer;
}

bool replacePrintfCalls(llvm::Function *OldPrintf, llvm::Function *NewPrintf) {
  PrintfCallsByFunction CallsByFunction = collectPrintfCalls(OldPrintf);
  llvm::Type *FormatType = OldPrintf->getFunctionType()->getParamType(0);
  for (auto &[Function, Calls] : CallsByFunction) {
    llvm::AllocaInst *ArgumentBuffer = createArgumentBuffer(Function, Calls);
    const CallLoweringContext Context = {NewPrintf, FormatType, ArgumentBuffer};
    for (llvm::CallInst *Call : Calls)
      if (!replaceCall(Call, Context))
        return false;
  }
  return true;
}

void lowerVPrintfAddressSpace(llvm::Module *Module) {
  llvm::Function *OldVPrintf = Module->getFunction("vprintf");
  if (!OldVPrintf)
    return;
  auto *OldFormatType = llvm::cast<llvm::PointerType>(
      OldVPrintf->getFunctionType()->getParamType(0));
  if (OldFormatType->getAddressSpace() == 0)
    return;

  llvm::LLVMContext &Context = Module->getContext();
  llvm::Type *Pointer = llvm::PointerType::get(Context, 0);
  auto *Type = llvm::FunctionType::get(OldVPrintf->getReturnType(),
                                       {Pointer, Pointer}, false);
  auto *NewVPrintf =
      llvm::Function::Create(Type, OldVPrintf->getLinkage(), "", Module);
  NewVPrintf->takeName(OldVPrintf);
  std::vector<llvm::User *> Users(OldVPrintf->user_begin(),
                                  OldVPrintf->user_end());
  for (llvm::User *User : Users) {
    auto *Call = llvm::dyn_cast<llvm::CallInst>(User);
    if (!Call)
      continue;
    llvm::IRBuilder<> Builder(Call);
    llvm::Value *Format = castPointerArgument(Builder, Call->getArgOperand(0));
    Call->setArgOperand(0, Format);
    Call->setCalledFunction(NewVPrintf);
  }
  OldVPrintf->eraseFromParent();
}

} // namespace

bool pocl_cuda_lower_printf(llvm::Module *Module) {
  llvm::Function *OldPrintf = Module->getFunction("printf");
  if (!OldPrintf)
    return true;
  if (OldPrintf->empty()) {
    POCL_MSG_ERR("[CUDA] printf declaration has no linked implementation\n");
    return false;
  }

  eraseVarArgIntrinsics(Module);
  llvm::LLVMContext &Context = Module->getContext();
  llvm::Type *FormatType = OldPrintf->getFunctionType()->getParamType(0);
  llvm::Type *ArgumentPointer = llvm::PointerType::get(Context, 0);
  auto *Type = llvm::FunctionType::get(OldPrintf->getReturnType(),
                                       {FormatType, ArgumentPointer}, false);
  auto *NewPrintf =
      llvm::Function::Create(Type, OldPrintf->getLinkage(), "", Module);
  NewPrintf->takeName(OldPrintf);
  NewPrintf->splice(NewPrintf->begin(), OldPrintf);

  llvm::AllocaInst *ArgumentIndex = createArgumentIndex(NewPrintf);
  replaceVaArgCalls(Module, NewPrintf, ArgumentIndex);
  if (!replacePrintfCalls(OldPrintf, NewPrintf))
    return false;

  llvm::Argument *OldFormat = OldPrintf->arg_begin();
  llvm::Argument *NewFormat = NewPrintf->arg_begin();
  NewFormat->takeName(OldFormat);
  OldFormat->replaceAllUsesWith(NewFormat);
  OldPrintf->eraseFromParent();
  lowerVPrintfAddressSpace(Module);
  return true;
}
