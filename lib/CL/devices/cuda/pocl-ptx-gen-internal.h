/* Internal interfaces shared by the CUDA PTX generation stages. */

#ifndef POCL_PTX_GEN_INTERNAL_H
#define POCL_PTX_GEN_INTERNAL_H

#include <map>
#include <string>
#include <vector>

namespace llvm {
class Module;
}

namespace pocl {
namespace cuda {

using AlignmentMap = std::map<std::string, std::vector<size_t>>;

bool verifyModule(llvm::Module *Module, const char *Step);

void addKernelAnnotations(llvm::Module *Module);
void fixConstantMemArgs(llvm::Module *Module);
void fixLocalMemArgs(llvm::Module *Module);
void handleGetWorkDim(llvm::Module *Module);
void handleGetGlobalOffset(llvm::Module *Module);
void handleNonUniformWorkGroup(llvm::Module *Module);

int linkLibDevice(llvm::Module *Module, const char *LibDevicePath,
                  bool FlushDenorms);
void mapLibDeviceCalls(llvm::Module *Module);

void createAlignmentMap(llvm::Module *Module, AlignmentMap *Map);

} // namespace cuda
} // namespace pocl

#endif
