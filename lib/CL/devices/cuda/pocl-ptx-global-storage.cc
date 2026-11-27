#include "pocl_cl.h"
#include "pocl-ptx-gen.h"

#include <limits>
#include <llvm/IR/Module.h>

extern "C" int
pocl_cuda_measure_global_storage (void *module, const void *opaque_device, size_t *bytes)
{
  const auto *device = static_cast<const _cl_device_id *> (opaque_device);
  *bytes = 0;
  if (device->global_var_max_size == 0)
    return CL_SUCCESS;
  const auto &ir = *static_cast<const llvm::Module *> (module);
  size_t total = 0;
  for (const auto &global : ir.globals ())
    {
      if (global.getAddressSpace () != device->global_as_id
          || global.isDeclaration ())
        continue;
      const auto size = ir.getDataLayout ().getTypeAllocSize (global.getValueType ());
      if (size.isScalable () || size.getFixedValue () > device->global_var_max_size
          || size.getFixedValue () > std::numeric_limits<size_t>::max () - total)
        {
          POCL_MSG_ERR ("CUDA program global '%s' exceeds the device storage limit\n",
                        global.getName ().str ().c_str ());
          return CL_BUILD_PROGRAM_FAILURE;
        }
      total += size.getFixedValue ();
    }
  *bytes = total;
  return CL_SUCCESS;
}
