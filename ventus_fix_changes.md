# Ventus Device Uninit Fix

## Problem
- pocl_ventus_uninit was not being called
- vt_dev_close was never executed
- Root cause: ops->reinit was NULL, causing skip in pocl_uninit_devices

## Solution
- Added pocl_ventus_reinit function implementation
- Updated ops->reinit = pocl_ventus_reinit
- Added function declaration in header file

## Files Modified
1. pocl/lib/CL/devices/ventus/pocl_ventus.cc
   - Line 132: Changed ops->reinit from NULL to pocl_ventus_reinit
   - Added pocl_ventus_reinit function implementation after line 990

2. pocl/lib/CL/devices/ventus/pocl_ventus.h
   - Added pocl_ventus_reinit function declaration

## Testing
- Set POCL_ENABLE_UNINIT=1
- Use GDB to verify pocl_ventus_uninit is called
- Confirm vt_dev_close is executed
