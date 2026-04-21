/*
 * chronos_ta.h — CHRONOS Trusted Application header.
 *
 * Defines the TA UUID, command identifiers, error codes, and sizing
 * constants shared between the Trusted Application (Secure World) and
 * the Normal World host wrapper.
 *
 * Matches the six GlobalPlatform TEE Internal Core API commands
 * described in Section 6.1 of the CHRONOS paper.
 */

#ifndef CHRONOS_TA_H
#define CHRONOS_TA_H

/* TA UUID — must match the BINARY line in the Makefile */
#define TA_CHRONOS_UUID \
    { 0x8a1b2c3d, 0x4e5f, 0x6a7b, \
      { 0x8c, 0x9d, 0x0e, 0x1f, 0x2a, 0x3b, 0x4c, 0x5d } }

/* ------------------------------------------------------------------ */
/*  Command IDs (invoked via TEE_InvokeCommandEntryPoint)              */
/* ------------------------------------------------------------------ */

/*
 * KEYGEN — Generate an ephemeral X25519 keypair.
 *   params[0].memref (out): 32-byte public key pk_i
 *   Returns: TEE_SUCCESS on success.
 *   Side-effect: sk_i is retained as a persistent object handle
 *               inside Secure World memory.
 */
#define TA_CHRONOS_CMD_KEYGEN               0

/*
 * COMPUTE_SEEDS — Derive pairwise DH secrets and encryption/PRG keys.
 *   params[0].memref (in):  Buffer of (N-1) x 32-byte peer public keys
 *   params[0].value.a:      Number of peers (N-1)
 *   Returns: TEE_SUCCESS on success.
 *   Side-effect: Stores raw DH secrets {s_{i,j}} and derived keys
 *               {k^enc_{i,j}, k^prg_{i,j}} in Secure World memory.
 */
#define TA_CHRONOS_CMD_COMPUTE_SEEDS        1

/*
 * SEAL(t) — Shamir-share sk_i, encrypt shares, seal seeds, init counter.
 *   params[0].value.a:      Shamir threshold t
 *   params[0].value.b:      Total number of peers (N-1)
 *   params[1].memref (out): Buffer for (N-1) encrypted shares
 *                           Each share: 32-byte ciphertext + 12-byte IV + 16-byte tag = 60 bytes
 *   params[2].value.a (out): Actual number of share ciphertexts written
 *   Returns: TEE_SUCCESS on success.
 *   Side-effect: Seeds sealed to Secure Storage under HUK,
 *               counter C initialised to 0, sk_i securely erased.
 */
#define TA_CHRONOS_CMD_SEAL                 2

/*
 * GENERATE_MASK(D, r) — Produce the round-r pseudorandom mask.
 *   params[0].value.a:      Requested round index r
 *   params[0].value.b:      Model dimension D (number of F_p elements)
 *   params[1].memref (out): D x uint32_t mask vector in [0, p-1]
 *   Returns: TEE_SUCCESS on success,
 *            TEE_ERROR_SECURITY if r <= C (rollback).
 *            CHRONOS_ERR_NOT_READY if seeds have not been sealed.
 */
#define TA_CHRONOS_CMD_GENERATE_MASK        3

/*
 * DECRYPT_SHARE(i, ciphertext) — Decrypt a Shamir share from peer i.
 *   params[0].value.a:      Peer index i (the dropped client)
 *   params[1].memref (in):  60-byte encrypted share ciphertext
 *   params[2].memref (out): 32-byte plaintext share sigma_{i->j}
 *   Returns: TEE_SUCCESS, or TEE_ERROR_MAC_INVALID on auth failure.
 */
#define TA_CHRONOS_CMD_DECRYPT_SHARE        4

/*
 * PEEK_COUNTER — Return the current monotonic round counter C.
 *   params[0].value.a (out): Current counter value C
 *   Returns: TEE_SUCCESS.
 *   Note: Read-only; does not modify state.
 */
#define TA_CHRONOS_CMD_PEEK_COUNTER         5

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

/* Mersenne prime p = 2^31 - 1, used for finite-field arithmetic */
#define CHRONOS_FIELD_PRIME     2147483647U

/* Maximum number of peers (N-1) supported */
#define CHRONOS_MAX_PEERS       63

/* Size of a single X25519 public/private key */
#define CHRONOS_KEY_SIZE        32

/* AES-128-GCM IV length for share encryption */
#define CHRONOS_GCM_IV_LEN     12

/* AES-128-GCM tag length */
#define CHRONOS_GCM_TAG_LEN    16

/* Total ciphertext size for one encrypted Shamir share:
 * 32 (share) + 12 (IV) + 16 (tag) = 60 bytes */
#define CHRONOS_SHARE_CT_SIZE  (CHRONOS_KEY_SIZE + CHRONOS_GCM_IV_LEN + CHRONOS_GCM_TAG_LEN)

/* RPMB object identifier for the persistent counter */
#define CHRONOS_RPMB_OBJ_ID    "chronos_round_counter"

/* Sealed seed-set object identifier in Secure Storage */
#define CHRONOS_SEEDS_OBJ_ID   "chronos_sealed_seeds"

/* ------------------------------------------------------------------ */
/*  Custom error codes                                                 */
/* ------------------------------------------------------------------ */
#define CHRONOS_ERR_NOT_READY   0xFFFF0100  /* SEAL has not been called */

#endif /* CHRONOS_TA_H */
