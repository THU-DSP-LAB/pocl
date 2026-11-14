/* Request-queue-owned migration queues, separated by source device. */
#ifndef POCL_MIGRATION_QUEUES_H
#define POCL_MIGRATION_QUEUES_H
#include "pocl_cl.h"
cl_int pocl_prepare_migration_queues (cl_command_queue queue);
cl_command_queue pocl_migration_queue (cl_command_queue queue, unsigned index);
void pocl_release_migration_queues (cl_command_queue queue);
#endif
