/*
 * The name of this file must not be modified
 */
#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

#include "chronos_ta.h"

#define TA_UUID TA_CHRONOS_UUID

/*
 * TA properties: multi-instance, keeps state between invocations (for counter).
 */
#define TA_FLAGS                    (TA_FLAG_USER_MODE | TA_FLAG_EXEC_DDR | TA_FLAG_INSTANCE_KEEP_ALIVE)

/* 
 * Memory footprint matching the paper's 18 KB Secure World claim:
 * 2 KB Stack, 16 KB Heap 
 */
#define TA_STACK_SIZE               (2 * 1024)
#define TA_DATA_SIZE                (16 * 1024)

#endif /* USER_TA_HEADER_DEFINES_H */
