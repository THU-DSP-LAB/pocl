/* test-ptx-scalarize.cc - regression test for NVPTX vector scalarization

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

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>

static const char *ComparisonIR = R"(
define ptx_kernel void @test_i16(ptr addrspace(1) %source,
                                 ptr addrspace(1) %destination) {
entry:
  %value = load <2 x i16>, ptr addrspace(1) %source, align 4
  %comparison = icmp eq <2 x i16> %value, zeroinitializer
  %result = sext <2 x i1> %comparison to <2 x i16>
  store <2 x i16> %result, ptr addrspace(1) %destination, align 4
  ret void
}

define ptx_kernel void @test_i8(ptr addrspace(1) %left,
                                ptr addrspace(1) %right,
                                ptr addrspace(1) %destination) {
entry:
  %left.value = load <2 x i8>, ptr addrspace(1) %left, align 2
  %right.value = load <2 x i8>, ptr addrspace(1) %right, align 2
  %comparison = icmp slt <2 x i8> %left.value, %right.value
  %result = sext <2 x i1> %comparison to <2 x i8>
  store <2 x i8> %result, ptr addrspace(1) %destination, align 2
  ret void
}

)";

static bool fail(const char *Message) {
  llvm::errs() << "test_cuda_scalarize: " << Message << '\n';
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
      llvm::parseAssemblyString(ComparisonIR, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print("test-ptx-scalarize", llvm::errs());
    return nullptr;
  }
  Module->setTargetTriple(Machine.getTargetTriple().str());
  Module->setDataLayout(Machine.createDataLayout());
  return Module;
}

static bool hasUnsupportedVectorInstruction(const llvm::Module &Module) {
  for (const llvm::Function &Function : Module)
    for (const llvm::BasicBlock &Block : Function)
      for (const llvm::Instruction &Instruction : Block)
        if (const auto *Compare = llvm::dyn_cast<llvm::CmpInst>(&Instruction)) {
          if (Compare->getOperand(0)->getType()->isVectorTy())
            return true;
        } else if (llvm::isa<llvm::LoadInst>(Instruction) ||
                   llvm::isa<llvm::StoreInst>(Instruction)) {
          if (Instruction.getType()->isVectorTy() ||
              Instruction.getOperand(0)->getType()->isVectorTy())
            return true;
        }
  return false;
}

static std::string emitPtx(llvm::Module &Module, llvm::TargetMachine &Machine) {
  llvm::SmallVector<char, 0> Buffer;
  llvm::raw_svector_ostream Output(Buffer);
  llvm::legacy::PassManager Passes;
  if (Machine.addPassesToEmitFile(Passes, Output, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile)) {
    fail("NVPTX target cannot emit assembly");
    return {};
  }
  Passes.run(Module);
  if (Buffer.empty()) {
    fail("NVPTX emitted empty assembly");
    return {};
  }
  return std::string(Buffer.begin(), Buffer.end());
}

static bool validatePtx(const std::string &Ptx) {
  if (Ptx.find(".v2.u8") != std::string::npos)
    return fail("NVPTX re-vectorized scalar i8 memory operations");
  if (Ptx.find("ld.volatile.global.s8") == std::string::npos)
    return fail("signed i8 comparison operands were not sign-extended");
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
  pocl_cuda_scalarize_vectors(Module.get());
  if (hasUnsupportedVectorInstruction(*Module)) {
    fail("v2i16 comparison or memory operation was not scalarized");
    return 1;
  }
  if (llvm::verifyModule(*Module, &llvm::errs()))
    return 1;
  const std::string Ptx = emitPtx(*Module, *Machine);
  return !Ptx.empty() && validatePtx(Ptx) ? 0 : 1;
}
