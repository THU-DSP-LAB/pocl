#ifndef POCL_LLVM_ADDRESS_SPACES_HH
#define POCL_LLVM_ADDRESS_SPACES_HH

#include <string>

namespace llvm {
class Module;
}

bool pocl_llvm_remap_address_space(llvm::Module &Module, unsigned Source,
                                  unsigned Target, std::string &Error);

#endif
