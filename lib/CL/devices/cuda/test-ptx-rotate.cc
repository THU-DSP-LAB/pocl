/* test-ptx-rotate.cc - regression test for NVPTX i64 rotate lowering.

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

static const char *RotateIR = R"(
define i64 @_Z10_cl_rotatell(i64 %value, i64 %amount) {
entry:
  %masked = and i64 %amount, 63
  %inverse = sub i64 0, %amount
  %inverse.masked = and i64 %inverse, 63
  %left = shl i64 %value, %masked
  %right = lshr i64 %value, %inverse.masked
  %result = or i64 %left, %right
  ret i64 %result
}

define <2 x i64> @_Z10_cl_rotateDv2_mS_(<2 x i64> %value,
                                         <2 x i64> %amount) {
entry:
  ret <2 x i64> zeroinitializer
}

define i64 @inlined_rotate(i64 %value, i64 %amount) {
entry:
  %masked = and i64 %amount, 63
  %inverse = sub i64 0, %amount
  %inverse.masked = and i64 %inverse, 63
  %left = shl i64 %value, %masked
  %right = lshr i64 %value, %inverse.masked
  %result = or i64 %left, %right
  ret i64 %result
}

define i64 @intrinsic_rotate(i64 %value, i64 %amount) {
entry:
  %result = call i64 @llvm.fshl.i64(i64 %value, i64 %value, i64 %amount)
  ret i64 %result
}

declare i64 @llvm.fshl.i64(i64, i64, i64)
)";

static bool fail(const char *Message) {
  llvm::errs() << "test_cuda_rotate: " << Message << '\n';
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
      llvm::parseAssemblyString(RotateIR, Diagnostic, Context);
  if (!Module) {
    Diagnostic.print("test-ptx-rotate", llvm::errs());
    return nullptr;
  }
  Module->setTargetTriple(Machine.getTargetTriple().str());
  Module->setDataLayout(Machine.createDataLayout());
  return Module;
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
  return std::string(Buffer.begin(), Buffer.end());
}

static size_t countOccurrences(const std::string &Text,
                               const std::string &Pattern) {
  size_t Count = 0;
  for (size_t Position = Text.find(Pattern); Position != std::string::npos;
       Position = Text.find(Pattern, Position + Pattern.size()))
    ++Count;
  return Count;
}

static bool validatePtx(const std::string &Ptx) {
  if (Ptx.find("%fd") != std::string::npos)
    return fail("i64 rotate used floating-point inline-asm registers");
  if (countOccurrences(Ptx, "and.b32") != 5)
    return fail("each scalar i64 rotate must mask its shift amount");
  if (countOccurrences(Ptx, "or.b64") != 5)
    return fail("scalar and vector i64 rotate bodies were not lowered");
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
  if (pocl_cuda_lower_i64_rotates(Module.get()) != 4)
    return fail("expected overload, expression, and intrinsic lowering") ? 0
                                                                         : 1;
  if (llvm::verifyModule(*Module, &llvm::errs()))
    return 1;
  const std::string Ptx = emitPtx(*Module, *Machine);
  return !Ptx.empty() && validatePtx(Ptx) ? 0 : 1;
}
