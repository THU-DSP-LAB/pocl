/* Internal, disjoint migration domains for buffer aliases. */
#ifndef POCL_BUFFER_ALIASES_H
#define POCL_BUFFER_ALIASES_H
#include "pocl_mem_management.h"

#define POCL_ALIAS_ACTIVE 1
#define POCL_ALIAS_RETIRED 2

pocl_buffer_migration_info *
pocl_lock_buffer_roots (pocl_buffer_migration_info *usage, cl_int *status);
void pocl_discard_alias_migrations (pocl_buffer_migration_info *migrations);
void pocl_unlock_buffer_roots (pocl_buffer_migration_info *roots);
pocl_buffer_migration_info *
pocl_buffer_alias_migrations (pocl_buffer_migration_info *usage,
                              cl_int *status);
void pocl_retain_actual_buffers (pocl_buffer_migration_info *actual,
                                 pocl_buffer_migration_info *migrations);
pocl_buffer_migration_info *
pocl_merge_actual_buffers (pocl_buffer_migration_info *actual,
                           pocl_buffer_migration_info *migrations);
#endif
