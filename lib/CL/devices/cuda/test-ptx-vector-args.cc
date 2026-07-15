/* test-ptx-vector-args.cc - regression test for CUDA vector argument lowering

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

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>

static const char *VectorKernelIR = R"(
define spir_kernel void @vector_args(<2 x i8> %a, <3 x i8> %b,
                                     <3 x i16> %c, i32 %d)
    !kernel_arg_access_qual !0 {
entry:
  %a.addr = alloca <2 x i8>
  %b.addr = alloca <3 x i8>
  %c.addr = alloca <3 x i16>
  store <2 x i8> %a, ptr %a.addr
  store <3 x i8> %b, ptr %b.addr
  store <3 x i16> %c, ptr %c.addr
  ret void
}
!0 = !{!"none", !"none", !"none", !"none"}
)";

static bool fail(const char *Message) {
  llvm::errs() << "test_cuda_vector_args: " << Message << '\n';
  return false;
}

static std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  std::string Error;
  const llvm::Target *Target
      = llvm::TargetRegistry::lookupTarget("nvptx64", Error);
  if (!Target) {
    llvm::errs() << Error << '\n';
    return nullptr;
  }
  llvm::TargetOptions Options;
  return std::unique_ptr<llvm::TargetMachine>(Target->createTargetMachine(
      "nvptx64-nvidia-cuda", "sm_89", "+ptx80", Options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOptLevel::Default));
}

static std::unique_ptr<llvm::Module>
parseModule(llvm::LLVMContext &Context, const llvm::TargetMachine &Machine) {
  llvm::SMDiagnostic Diagnostic;
  std::unique_ptr<llvm::Module> Module
      = llvm::parseAssemblyString(VectorKernelIR, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print("test-ptx-vector-args", llvm::errs());
    return nullptr;
  }
  Module->setTargetTriple(Machine.getTargetTriple().str());
  Module->setDataLayout(Machine.createDataLayout());
  return Module;
}

static bool validateKernel(const llvm::Function &Kernel) {
  const llvm::DataLayout &Layout = Kernel.getParent()->getDataLayout();
  unsigned Index = 0;
  unsigned VectorCount = 0;
  for (const auto &Argument : Kernel.args()) {
    llvm::Type *ByValType = Kernel.getParamByValType(Index);
    if (Index == 3) {
      if (!Argument.getType()->isIntegerTy(32) || ByValType)
        return fail("scalar argument changed");
    } else {
      if (!Argument.getType()->isPointerTy() || !ByValType
          || !ByValType->isVectorTy())
        return fail("vector argument was not converted to byval pointer");
      if (Kernel.getParamAlign(Index) != Layout.getABITypeAlign(ByValType))
        return fail("byval vector alignment is incorrect");
      ++VectorCount;
    }
    ++Index;
  }
  return (Index == 4 && VectorCount == 3)
             ? true
             : fail("unexpected transformed argument count");
}

static bool emitPtx(llvm::Module &Module, llvm::TargetMachine &Machine) {
  llvm::SmallVector<char, 0> Buffer;
  llvm::raw_svector_ostream Output(Buffer);
  llvm::legacy::PassManager Passes;
  if (Machine.addPassesToEmitFile(Passes, Output, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile))
    return fail("NVPTX target cannot emit assembly");
  Passes.run(Module);
  if (!llvm::StringRef(Buffer.data(), Buffer.size()).contains(".align 8 .b8"))
    return fail("PTX does not contain the aligned byval parameter");
  return true;
}

int main() {
  std::unique_ptr<llvm::TargetMachine> Machine = createTargetMachine();
  if (!Machine)
    return 1;
  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> Module = parseModule(Context, *Machine);
  if (!Module)
    return 1;

  pocl_cuda_fix_vector_args(Module.get());
  llvm::Function *Kernel = Module->getFunction("vector_args");
  if (!Kernel) {
    fail("transformed kernel is missing");
    return 1;
  }
  if (!validateKernel(*Kernel)) {
    Module->print(llvm::errs(), nullptr);
    return 1;
  }
  if (llvm::verifyModule(*Module, &llvm::errs()))
    return 1;
  return emitPtx(*Module, *Machine) ? 0 : 1;
}
