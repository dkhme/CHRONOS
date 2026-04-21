#include <err.h>
#include <stdio.h>
#include <string.h>
#include <tee_client_api.h>
#include "chronos_ta.h"

/* 
 * TEEC C-Wrapper for Python (ctypes).
 * Handles the Normal World/Secure World context switching and shared memory.
 */

static TEEC_Context ctx;
static TEEC_Session sess;
static int initialized = 0;

int chronos_tee_init(void) {
    TEEC_UUID uuid = TA_CHRONOS_UUID;
    TEEC_Result res;
    uint32_t err_origin;

    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS) return -1;

    res = TEEC_OpenSession(&ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS) {
        TEEC_FinalizeContext(&ctx);
        return -1;
    }

    initialized = 1;
    return 0;
}

void chronos_tee_close(void) {
    if (initialized) {
        TEEC_CloseSession(&sess);
        TEEC_FinalizeContext(&ctx);
        initialized = 0;
    }
}

/*
 * Invokes the TA to generate the mask. 
 * Expected to be called via Python ctypes.
 */
int chronos_generate_mask(uint32_t round_id, uint32_t dimension, uint32_t* out_buffer) {
    if (!initialized) {
        if (chronos_tee_init() != 0) return -1;
    }

    TEEC_Operation op;
    TEEC_SharedMemory shm;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));

    // Allocate Shared Memory for the mask vector
    shm.size = dimension * sizeof(uint32_t);
    shm.flags = TEEC_MEM_OUTPUT;
    res = TEEC_AllocateSharedMemory(&ctx, &shm);
    if (res != TEEC_SUCCESS) return -2;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_MEMREF_PARTIAL_OUTPUT, TEEC_NONE, TEEC_NONE);
    
    // Parameter 0: Round ID and Dimension
    op.params[0].value.a = round_id;
    op.params[0].value.b = dimension;
    
    // Parameter 1: Shared Memory Buffer for Mask Output
    op.params[1].memref.parent = &shm;
    op.params[1].memref.offset = 0;
    op.params[1].memref.size = shm.size;

    // Invoke TA command
    res = TEEC_InvokeCommand(&sess, TA_CHRONOS_CMD_GENERATE_MASK, &op, &err_origin);
    
    if (res == TEEC_SUCCESS) {
        // Copy out of shared memory into the Python-provided buffer
        memcpy(out_buffer, shm.buffer, shm.size);
    }

    TEEC_ReleaseSharedMemory(&shm);

    if (res == CHRONOS_ERR_NOT_READY) return -3;
    // Explicit rollback detection check
    if (res == TEE_ERROR_SECURITY) return -4; 
    
    return (res == TEEC_SUCCESS) ? 0 : -1;
}

/*
 * Invokes the TA to peek at the current round counter.
 * Expected to be called via Python ctypes.
 */
int chronos_peek_counter(uint32_t* out_counter) {
    if (!initialized) {
        if (chronos_tee_init() != 0) return -1;
    }

    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_OUTPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);

    res = TEEC_InvokeCommand(&sess, TA_CHRONOS_CMD_PEEK_COUNTER, &op, &err_origin);

    if (res == TEEC_SUCCESS) {
        *out_counter = op.params[0].value.a;
        return 0;
    }

    if (res == CHRONOS_ERR_NOT_READY) {
        return -3; // ERR_NOT_READY
    }

    return -1;
}