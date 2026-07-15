/* test-ptx-printf.cc - regression test for CUDA printf ABI lowering

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

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>

static const char *PrintfIR = R"(
@format = addrspace(4) constant [14 x i8] c"%v4hlf %v16ld\00"

declare void @__cl_va_arg(ptr, ptr, i32)
declare i32 @vprintf(ptr addrspace(4), ptr)

define i32 @printf(ptr addrspace(4) %format, ...) {
entry:
  %data = alloca [128 x i8], align 16
  call void @__cl_va_arg(ptr undef, ptr %data, i32 32)
  call void @__cl_va_arg(ptr undef, ptr %data, i32 32)
  %result = call i32 @vprintf(ptr addrspace(4) %format, ptr %data)
  ret i32 %result
}

define ptx_kernel void @printf_kernel() {
entry:
  %result = call i32 (ptr addrspace(4), ...) @printf(
      ptr addrspace(4) @format,
      <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>,
      <16 x i64> zeroinitializer)
  %second = call i32 (ptr addrspace(4), ...) @printf(
      ptr addrspace(4) @format, i32 42)
  ret void
}
)";

static bool fail(const char *Message) {
  llvm::errs() << "test_cuda_printf: " << Message << '\n';
  return false;
}

static std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  std::string Error;
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget("nvptx64", Error);
  if (!Target) {
    llvm::errs() << Error << '\n';
    return nullptr;
  }
  llvm::TargetOptions Options;
  return std::unique_ptr<llvm::TargetMachine>(Target->createTargetMachine(
      "nvptx64-nvidia-cuda", "sm_89", "+ptx80", Options, llvm::Reloc::PIC_,
      llvm::CodeModel::Small, llvm::CodeGenOptLevel::Default));
}

static std::unique_ptr<llvm::Module>
parseModule(llvm::LLVMContext &Context, const llvm::TargetMachine &Machine) {
  llvm::SMDiagnostic Diagnostic;
  std::unique_ptr<llvm::Module> Module =
      llvm::parseAssemblyString(PrintfIR, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print("test-ptx-printf", llvm::errs());
    return nullptr;
  }
  Module->setTargetTriple(Machine.getTargetTriple().str());
  Module->setDataLayout(Machine.createDataLayout());
  return Module;
}

static bool validatePrintf(const llvm::Module &Module) {
  const llvm::Function *Printf = Module.getFunction("printf");
  if (!Printf || Printf->isVarArg() || Printf->arg_size() != 2)
    return fail("printf did not become a two-argument function");
  unsigned Copies = 0;
  for (const llvm::BasicBlock &Block : *Printf)
    for (const llvm::Instruction &Instruction : Block)
      if (const auto *Copy = llvm::dyn_cast<llvm::MemCpyInst>(&Instruction)) {
        const auto *Length =
            llvm::dyn_cast<llvm::ConstantInt>(Copy->getLength());
        Copies += Length && Length->getZExtValue() == 128;
      }
  return Copies == 2 ? true : fail("va_arg did not copy two 128-byte slots");
}

static bool validateKernel(const llvm::Module &Module) {
  const llvm::Function *Kernel = Module.getFunction("printf_kernel");
  if (!Kernel)
    return fail("printf kernel is missing");
  const llvm::Value *ArgumentBuffer = nullptr;
  unsigned Allocations = 0;
  unsigned Calls = 0;
  for (const llvm::BasicBlock &Block : *Kernel)
    for (const llvm::Instruction &Instruction : Block) {
      if (const auto *Allocation =
              llvm::dyn_cast<llvm::AllocaInst>(&Instruction)) {
        if (!Allocation->getAllocatedType()->isIntegerTy(8))
          continue;
        const auto *Size =
            llvm::dyn_cast<llvm::ConstantInt>(Allocation->getArraySize());
        if (!Size || Size->getZExtValue() != 256 ||
            Allocation->getAlign().value() != 128)
          return fail("kernel argument buffer has the wrong layout");
        ArgumentBuffer = Allocation;
        ++Allocations;
      } else if (const auto *Call =
                     llvm::dyn_cast<llvm::CallInst>(&Instruction)) {
        if (Call->getCalledFunction() != Module.getFunction("printf"))
          continue;
        if (Call->getArgOperand(1) != ArgumentBuffer)
          return fail("printf calls do not share their argument buffer");
        ++Calls;
      }
    }
  return Allocations == 1 && Calls == 2
             ? true
             : fail("kernel does not contain one shared argument buffer");
}

static bool emitPtx(llvm::Module &Module, llvm::TargetMachine &Machine) {
  llvm::SmallVector<char, 0> Buffer;
  llvm::raw_svector_ostream Output(Buffer);
  llvm::legacy::PassManager Passes;
  if (Machine.addPassesToEmitFile(Passes, Output, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile))
    return fail("NVPTX target cannot emit assembly");
  Passes.run(Module);
  return llvm::StringRef(Buffer.data(), Buffer.size()).contains("vprintf")
             ? true
             : fail("PTX does not contain the vprintf system call");
}

int main() {
  std::unique_ptr<llvm::TargetMachine> Machine = createTargetMachine();
  if (!Machine)
    return 1;
  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> Module = parseModule(Context, *Machine);
  if (!Module || !pocl_cuda_lower_printf(Module.get()))
    return 1;
  if (!validatePrintf(*Module) || !validateKernel(*Module)) {
    Module->print(llvm::errs(), nullptr);
    return 1;
  }
  if (llvm::verifyModule(*Module, &llvm::errs()))
    return 1;
  return emitPtx(*Module, *Machine) ? 0 : 1;
}
