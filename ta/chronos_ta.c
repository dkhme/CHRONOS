/*
 * chronos_ta.c — CHRONOS Trusted Application for OP-TEE.
 *
 * Implements the six GlobalPlatform TEE Internal Core API commands
 * described in Section 6.1 of the CHRONOS paper:
 *
 *   KEYGEN          — Generate ephemeral X25519 keypair; retain sk_i
 *   COMPUTE_SEEDS   — Derive pairwise DH secrets and AES keys via HKDF
 *   SEAL            — Shamir-share sk_i, encrypt shares, seal seeds,
 *                     initialise RPMB-backed counter, erase sk_i
 *   GENERATE_MASK   — AES-128-CTR mask with rejection sampling in F_p
 *   DECRYPT_SHARE   — Decrypt a peer's Shamir share for dropout recovery
 *   PEEK_COUNTER    — Return the current round counter (read-only)
 *
 * Build requirements:
 *   OP-TEE 4.4.0+, CFG_CRYPTO_X25519=y, CFG_WITH_VFP=y
 *   ARM Cortex-A72 (RK3399 / Rock Pi 4) with ARMv8 Crypto Extensions.
 *
 * Memory budget (persistent Secure World storage):
 *   (N-1) x 32 bytes  (DH secrets)
 *   + 4 bytes          (round counter)
 *   + 16 bytes         (AES-256-GCM tag for sealed blob)
 *   ≈ 1012 bytes for N = 32
 */

#include <string.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include "chronos_ta.h"

/* ================================================================== */
/*  Compile-time configuration                                         */
/* ================================================================== */

/* Default client index — overridden at provisioning time via a
 * persistent object; kept as a compile-time fallback for testing. */
#ifndef CHRONOS_CLIENT_INDEX
#define CHRONOS_CLIENT_INDEX   0
#endif

/* ================================================================== */
/*  Per-session state (lives only in Secure World RAM)                  */
/* ================================================================== */

/* Ephemeral X25519 private key — exists only between KEYGEN and SEAL */
static uint8_t  sk_i[CHRONOS_KEY_SIZE];
static int      sk_valid = 0;

/* Raw Diffie-Hellman shared secrets s_{i,j} for each peer */
static uint8_t  dh_secrets[CHRONOS_MAX_PEERS][CHRONOS_KEY_SIZE];

/* Derived 128-bit AES-GCM encryption keys k^enc_{i,j} */
static uint8_t  enc_keys[CHRONOS_MAX_PEERS][16];

/* Derived 128-bit AES-CTR PRG keys k^prg_{i,j} */
static uint8_t  prg_keys[CHRONOS_MAX_PEERS][16];

/* Number of established peers (N-1) */
static uint32_t num_peers = 0;

/* Whether SEAL has completed (seeds are in Secure Storage) */
static int      sealed = 0;

/* Hardware-backed monotonic round counter C */
static uint32_t round_counter = 0;

/* Client index within the federation */
static uint32_t client_index = CHRONOS_CLIENT_INDEX;

/* ================================================================== */
/*  Helpers: HKDF-SHA256 one-shot (info-string differentiation)        */
/* ================================================================== */

/*
 * HKDF-SHA256 extract-then-expand (RFC 5869), single-block output.
 *
 *   Extract:  PRK  = HMAC-SHA256(salt, IKM)          [salt = 32 zero bytes]
 *   Expand:   OKM  = HMAC-SHA256(PRK, info ‖ 0x01)   [single block ≤ 32 bytes]
 *
 * The 'info' string differentiates enc vs. prg keys ("chronos-enc" / "chronos-prg").
 * Requires out_len ≤ 32.
 */
static TEE_Result hkdf_sha256(const uint8_t *ikm, uint32_t ikm_len,
                              const char *info, uint32_t info_len,
                              uint8_t *out, uint32_t out_len)
{
    TEE_Result res;
    TEE_OperationHandle op = TEE_HANDLE_NULL;
    TEE_ObjectHandle key_obj = TEE_HANDLE_NULL;
    TEE_Attribute attr;
    uint8_t prk[32];
    uint32_t mac_len;

    /* ---- Extract phase: PRK = HMAC-SHA256(salt, IKM) ---- */
    static const uint8_t zero_salt[32] = {0};

    /* Create HMAC key from salt (32 zero bytes per RFC 5869 default) */
    res = TEE_AllocateTransientObject(TEE_TYPE_HMAC_SHA256, 256, &key_obj);
    if (res != TEE_SUCCESS) return res;

    TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE,
                         (void *)zero_salt, sizeof(zero_salt));
    res = TEE_PopulateTransientObject(key_obj, &attr, 1);
    if (res != TEE_SUCCESS) goto out_key;

    res = TEE_AllocateOperation(&op, TEE_ALG_HMAC_SHA256, TEE_MODE_MAC, 256);
    if (res != TEE_SUCCESS) goto out_key;

    TEE_SetOperationKey(op, key_obj);
    TEE_MACInit(op, NULL, 0);

    mac_len = sizeof(prk);
    res = TEE_MACComputeFinal(op, ikm, ikm_len, prk, &mac_len);
    TEE_FreeOperation(op);
    op = TEE_HANDLE_NULL;
    TEE_FreeTransientObject(key_obj);
    key_obj = TEE_HANDLE_NULL;
    if (res != TEE_SUCCESS) return res;

    /* ---- Expand phase: T(1) = HMAC-SHA256(PRK, info ‖ 0x01) ---- */
    res = TEE_AllocateTransientObject(TEE_TYPE_HMAC_SHA256, 256, &key_obj);
    if (res != TEE_SUCCESS) return res;

    TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, prk, sizeof(prk));
    res = TEE_PopulateTransientObject(key_obj, &attr, 1);
    if (res != TEE_SUCCESS) goto out_key;

    res = TEE_AllocateOperation(&op, TEE_ALG_HMAC_SHA256, TEE_MODE_MAC, 256);
    if (res != TEE_SUCCESS) goto out_key;

    TEE_SetOperationKey(op, key_obj);
    TEE_MACInit(op, NULL, 0);
    TEE_MACUpdate(op, (const uint8_t *)info, info_len);

    uint8_t counter_byte = 0x01;
    uint8_t okm[32];
    mac_len = sizeof(okm);
    res = TEE_MACComputeFinal(op, &counter_byte, 1, okm, &mac_len);
    if (res != TEE_SUCCESS) goto out_op;

    /* Truncate to requested length */
    memcpy(out, okm, out_len < 32 ? out_len : 32);

    /* Wipe intermediate key material */
    TEE_MemFill(prk, 0, sizeof(prk));
    TEE_MemFill(okm, 0, sizeof(okm));

out_op:
    if (op != TEE_HANDLE_NULL)
        TEE_FreeOperation(op);
out_key:
    if (key_obj != TEE_HANDLE_NULL)
        TEE_FreeTransientObject(key_obj);
    return res;
}

/* ================================================================== */
/*  Helpers: RPMB-backed persistent counter                            */
/* ================================================================== */

/*
 * Reads the round counter from RPMB-backed Secure Storage.
 * If the object does not exist (first boot), initialises to 0.
 */
static TEE_Result rpmb_read_counter(uint32_t *value)
{
    TEE_ObjectHandle obj = TEE_HANDLE_NULL;
    TEE_Result res;

    res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE_RPMB,
                                   CHRONOS_RPMB_OBJ_ID,
                                   strlen(CHRONOS_RPMB_OBJ_ID),
                                   TEE_DATA_FLAG_ACCESS_READ,
                                   &obj);
    if (res == TEE_ERROR_ITEM_NOT_FOUND) {
        *value = 0;
        return TEE_SUCCESS;
    }
    if (res != TEE_SUCCESS) return res;

    uint32_t count = sizeof(*value);
    res = TEE_ReadObjectData(obj, value, sizeof(*value), &count);
    TEE_CloseObject(obj);
    return res;
}

/*
 * Writes (overwrites) the round counter to RPMB-backed Secure Storage.
 * This is a synchronous eMMC flush — ≈39 ms on RK3399.
 */
static TEE_Result rpmb_write_counter(uint32_t value)
{
    TEE_ObjectHandle obj = TEE_HANDLE_NULL;
    TEE_Result res;

    /* Try to open existing object */
    res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE_RPMB,
                                   CHRONOS_RPMB_OBJ_ID,
                                   strlen(CHRONOS_RPMB_OBJ_ID),
                                   TEE_DATA_FLAG_ACCESS_WRITE,
                                   &obj);
    if (res == TEE_ERROR_ITEM_NOT_FOUND) {
        /* First write — create the object */
        res = TEE_CreatePersistentObject(TEE_STORAGE_PRIVATE_RPMB,
                                         CHRONOS_RPMB_OBJ_ID,
                                         strlen(CHRONOS_RPMB_OBJ_ID),
                                         TEE_DATA_FLAG_ACCESS_WRITE,
                                         TEE_HANDLE_NULL, NULL, 0, &obj);
        if (res != TEE_SUCCESS) return res;
    } else if (res != TEE_SUCCESS) {
        return res;
    }

    res = TEE_WriteObjectData(obj, &value, sizeof(value));
    TEE_CloseObject(obj);
    return res;
}

/* ================================================================== */
/*  Helpers: Secure Storage for sealed seed set                        */
/* ================================================================== */

/*
 * Seal the DH secret array to Secure Storage under the device HUK.
 * OP-TEE encrypts persistent objects with an AES-256-GCM key derived
 * from the device Hardware Unique Key automatically.
 */
static TEE_Result seal_seeds(const uint8_t *data, uint32_t data_len)
{
    TEE_ObjectHandle obj = TEE_HANDLE_NULL;
    TEE_Result res;

    /* Remove any previous sealed blob */
    TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
                             CHRONOS_SEEDS_OBJ_ID,
                             strlen(CHRONOS_SEEDS_OBJ_ID),
                             TEE_DATA_FLAG_ACCESS_WRITE_META, &obj);
    if (obj != TEE_HANDLE_NULL) {
        TEE_CloseAndDeletePersistentObject1(obj);
        obj = TEE_HANDLE_NULL;
    }

    res = TEE_CreatePersistentObject(TEE_STORAGE_PRIVATE,
                                     CHRONOS_SEEDS_OBJ_ID,
                                     strlen(CHRONOS_SEEDS_OBJ_ID),
                                     TEE_DATA_FLAG_ACCESS_WRITE |
                                     TEE_DATA_FLAG_ACCESS_READ,
                                     TEE_HANDLE_NULL, data, data_len, &obj);
    if (res == TEE_SUCCESS)
        TEE_CloseObject(obj);
    return res;
}

/*
 * Unseal (read) the DH secret array from Secure Storage.
 */
static TEE_Result unseal_seeds(uint8_t *data, uint32_t data_len)
{
    TEE_ObjectHandle obj = TEE_HANDLE_NULL;
    TEE_Result res;
    uint32_t count = data_len;

    res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
                                   CHRONOS_SEEDS_OBJ_ID,
                                   strlen(CHRONOS_SEEDS_OBJ_ID),
                                   TEE_DATA_FLAG_ACCESS_READ,
                                   &obj);
    if (res != TEE_SUCCESS) return res;

    res = TEE_ReadObjectData(obj, data, data_len, &count);
    TEE_CloseObject(obj);
    return res;
}

/* ================================================================== */
/*  Command: KEYGEN                                                    */
/* ================================================================== */

static TEE_Result cmd_keygen(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;
    TEE_ObjectHandle keypair = TEE_HANDLE_NULL;

    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (params[0].memref.size < CHRONOS_KEY_SIZE)
        return TEE_ERROR_SHORT_BUFFER;

    /* Generate X25519 keypair inside the TEE */
    res = TEE_AllocateTransientObject(TEE_TYPE_X25519_KEYPAIR, 256, &keypair);
    if (res != TEE_SUCCESS) return res;

    res = TEE_GenerateKey(keypair, 256, NULL, 0);
    if (res != TEE_SUCCESS) goto out;

    /* Extract private key into Secure World buffer */
    uint32_t sk_len = CHRONOS_KEY_SIZE;
    res = TEE_GetObjectBufferAttribute(keypair, TEE_ATTR_X25519_PRIVATE_VALUE,
                                       sk_i, &sk_len);
    if (res != TEE_SUCCESS) goto out;
    sk_valid = 1;

    /* Export only the public key to Normal World */
    uint32_t pk_len = CHRONOS_KEY_SIZE;
    res = TEE_GetObjectBufferAttribute(keypair, TEE_ATTR_X25519_PUBLIC_VALUE,
                                       params[0].memref.buffer, &pk_len);

out:
    TEE_FreeTransientObject(keypair);
    return res;
}

/* ================================================================== */
/*  Command: COMPUTE_SEEDS                                             */
/* ================================================================== */

static TEE_Result cmd_compute_seeds(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;

    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                       TEE_PARAM_TYPE_MEMREF_INPUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (!sk_valid)
        return TEE_ERROR_BAD_STATE;

    uint32_t n_peers = params[0].value.a;
    if (n_peers > CHRONOS_MAX_PEERS)
        return TEE_ERROR_BAD_PARAMETERS;

    const uint8_t *peer_pks = (const uint8_t *)params[1].memref.buffer;
    if (params[1].memref.size < n_peers * CHRONOS_KEY_SIZE)
        return TEE_ERROR_SHORT_BUFFER;

    num_peers = n_peers;

    for (uint32_t j = 0; j < n_peers; j++) {
        /* Build a transient X25519 keypair object from our sk_i */
        TEE_ObjectHandle my_key = TEE_HANDLE_NULL;
        TEE_Attribute attrs[1];

        res = TEE_AllocateTransientObject(TEE_TYPE_X25519_KEYPAIR, 256, &my_key);
        if (res != TEE_SUCCESS) return res;

        TEE_InitRefAttribute(&attrs[0], TEE_ATTR_X25519_PRIVATE_VALUE,
                             sk_i, CHRONOS_KEY_SIZE);
        res = TEE_PopulateTransientObject(my_key, attrs, 1);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(my_key);
            return res;
        }

        /* Derive shared secret: s_{i,j} = DH(sk_i, pk_j) */
        TEE_OperationHandle dh_op = TEE_HANDLE_NULL;
        res = TEE_AllocateOperation(&dh_op, TEE_ALG_X25519, TEE_MODE_DERIVE, 256);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(my_key);
            return res;
        }

        TEE_SetOperationKey(dh_op, my_key);

        TEE_Attribute dh_attr;
        TEE_InitRefAttribute(&dh_attr, TEE_ATTR_X25519_PUBLIC_VALUE,
                             (void *)(peer_pks + j * CHRONOS_KEY_SIZE),
                             CHRONOS_KEY_SIZE);

        TEE_ObjectHandle derived = TEE_HANDLE_NULL;
        res = TEE_AllocateTransientObject(TEE_TYPE_GENERIC_SECRET, 256, &derived);
        if (res != TEE_SUCCESS) {
            TEE_FreeOperation(dh_op);
            TEE_FreeTransientObject(my_key);
            return res;
        }

        res = TEE_DeriveKey(dh_op, &dh_attr, 1, derived);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(derived);
            TEE_FreeOperation(dh_op);
            TEE_FreeTransientObject(my_key);
            return res;
        }

        /* Extract raw DH secret */
        uint32_t sec_len = CHRONOS_KEY_SIZE;
        res = TEE_GetObjectBufferAttribute(derived, TEE_ATTR_SECRET_VALUE,
                                           dh_secrets[j], &sec_len);

        TEE_FreeTransientObject(derived);
        TEE_FreeOperation(dh_op);
        TEE_FreeTransientObject(my_key);

        if (res != TEE_SUCCESS) return res;

        /* Derive k^enc_{i,j} = HKDF(s_{i,j}, info="chronos-enc") */
        res = hkdf_sha256(dh_secrets[j], CHRONOS_KEY_SIZE,
                          "chronos-enc", 11, enc_keys[j], 16);
        if (res != TEE_SUCCESS) return res;

        /* Derive k^prg_{i,j} = HKDF(s_{i,j}, info="chronos-prg") */
        res = hkdf_sha256(dh_secrets[j], CHRONOS_KEY_SIZE,
                          "chronos-prg", 11, prg_keys[j], 16);
        if (res != TEE_SUCCESS) return res;
    }

    return TEE_SUCCESS;
}

/* ================================================================== */
/*  Command: SEAL(t)                                                   */
/* ================================================================== */

/*
 * Shamir secret sharing over byte-wise GF(2^8).
 * Produces (n_peers) shares of the 32-byte secret sk_i,
 * each share being a 32-byte element (one byte-wise evaluation).
 *
 * For each byte position b in [0..31]:
 *   - Random polynomial f_b(x) of degree (t-1) with f_b(0) = sk_i[b]
 *   - Share for peer j: sigma_{i->j}[b] = f_b(j+1) evaluated in GF(2^8)
 */

/* GF(2^8) multiplication using the irreducible polynomial x^8 + x^4 + x^3 + x + 1 */
static uint8_t gf256_mul(uint8_t a, uint8_t b)
{
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1)
            result ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi)
            a ^= 0x1B; /* x^8 + x^4 + x^3 + x + 1 */
        b >>= 1;
    }
    return result;
}

/* Evaluate polynomial at point x in GF(2^8) */
static uint8_t gf256_poly_eval(const uint8_t *coeffs, uint32_t degree, uint8_t x)
{
    uint8_t result = coeffs[degree];
    for (int i = (int)degree - 1; i >= 0; i--)
        result = gf256_mul(result, x) ^ coeffs[i];
    return result;
}

static TEE_Result cmd_seal(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;

    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                       TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                       TEE_PARAM_TYPE_VALUE_OUTPUT,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (!sk_valid || num_peers == 0)
        return TEE_ERROR_BAD_STATE;

    uint32_t threshold = params[0].value.a;
    uint32_t n_peers   = params[0].value.b;

    if (n_peers > num_peers || threshold < 1 || threshold > n_peers)
        return TEE_ERROR_BAD_PARAMETERS;

    if (params[1].memref.size < n_peers * CHRONOS_SHARE_CT_SIZE)
        return TEE_ERROR_SHORT_BUFFER;

    uint8_t *out_buf = (uint8_t *)params[1].memref.buffer;

    /* ---- 1. Generate Shamir shares of sk_i ---- */
    uint8_t shares[CHRONOS_MAX_PEERS][CHRONOS_KEY_SIZE];
    uint8_t poly_coeffs[CHRONOS_MAX_PEERS]; /* max degree = MAX_PEERS-1 */

    for (uint32_t b = 0; b < CHRONOS_KEY_SIZE; b++) {
        /* Build random polynomial: coeffs[0] = sk_i[b], rest random */
        poly_coeffs[0] = sk_i[b];
        TEE_GenerateRandom(&poly_coeffs[1], threshold - 1);

        for (uint32_t j = 0; j < n_peers; j++) {
            /* Evaluate at x = j+1 (x=0 would reveal the secret) */
            shares[j][b] = gf256_poly_eval(poly_coeffs, threshold - 1,
                                            (uint8_t)(j + 1));
        }
    }

    /* ---- 2. Encrypt each share under k^enc_{i,j} with AES-128-GCM ---- */
    for (uint32_t j = 0; j < n_peers; j++) {
        uint8_t *ct_slot = out_buf + j * CHRONOS_SHARE_CT_SIZE;
        uint8_t iv[CHRONOS_GCM_IV_LEN];
        TEE_GenerateRandom(iv, CHRONOS_GCM_IV_LEN);

        /* Build AES-128-GCM key object */
        TEE_ObjectHandle aes_key = TEE_HANDLE_NULL;
        TEE_Attribute attr;
        res = TEE_AllocateTransientObject(TEE_TYPE_AES, 128, &aes_key);
        if (res != TEE_SUCCESS) return res;

        TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, enc_keys[j], 16);
        res = TEE_PopulateTransientObject(aes_key, &attr, 1);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(aes_key);
            return res;
        }

        TEE_OperationHandle ae_op = TEE_HANDLE_NULL;
        res = TEE_AllocateOperation(&ae_op, TEE_ALG_AES_GCM, TEE_MODE_ENCRYPT, 128);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(aes_key);
            return res;
        }

        TEE_SetOperationKey(ae_op, aes_key);
        res = TEE_AEInit(ae_op, iv, CHRONOS_GCM_IV_LEN,
                         CHRONOS_GCM_TAG_LEN * 8, 0, CHRONOS_KEY_SIZE);
        if (res != TEE_SUCCESS) {
            TEE_FreeOperation(ae_op);
            TEE_FreeTransientObject(aes_key);
            return res;
        }

        /* Encrypt: share → ciphertext (32 bytes) + tag (16 bytes) */
        uint8_t ciphertext[CHRONOS_KEY_SIZE];
        uint32_t ct_len = CHRONOS_KEY_SIZE;
        uint8_t tag[CHRONOS_GCM_TAG_LEN];
        uint32_t tag_len = CHRONOS_GCM_TAG_LEN;

        res = TEE_AEEncryptFinal(ae_op, shares[j], CHRONOS_KEY_SIZE,
                                  ciphertext, &ct_len, tag, &tag_len);

        TEE_FreeOperation(ae_op);
        TEE_FreeTransientObject(aes_key);

        if (res != TEE_SUCCESS) return res;

        /* Pack: [ciphertext(32) | iv(12) | tag(16)] */
        memcpy(ct_slot, ciphertext, CHRONOS_KEY_SIZE);
        memcpy(ct_slot + CHRONOS_KEY_SIZE, iv, CHRONOS_GCM_IV_LEN);
        memcpy(ct_slot + CHRONOS_KEY_SIZE + CHRONOS_GCM_IV_LEN, tag, CHRONOS_GCM_TAG_LEN);
    }

    params[2].value.a = n_peers;

    /* ---- 3. Seal DH secrets to Secure Storage under HUK ---- */
    res = seal_seeds((const uint8_t *)dh_secrets, n_peers * CHRONOS_KEY_SIZE);
    if (res != TEE_SUCCESS) return res;

    /* ---- 4. Initialise RPMB counter to 0 ---- */
    round_counter = 0;
    res = rpmb_write_counter(0);
    if (res != TEE_SUCCESS) return res;

    /* ---- 5. Securely erase sk_i from Secure World memory ---- */
    TEE_MemFill(sk_i, 0, CHRONOS_KEY_SIZE);
    sk_valid = 0;
    sealed = 1;

    return TEE_SUCCESS;
}

/* ================================================================== */
/*  Command: GENERATE_MASK(D, r)                                       */
/* ================================================================== */

static TEE_Result cmd_generate_mask(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;

    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                       TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (!sealed)
        return CHRONOS_ERR_NOT_READY;

    uint32_t requested_round = params[0].value.a;
    uint32_t dimension_D     = params[0].value.b;
    uint32_t *mask_buffer    = (uint32_t *)params[1].memref.buffer;

    if (params[1].memref.size < dimension_D * sizeof(uint32_t))
        return TEE_ERROR_SHORT_BUFFER;

    /* ---- 1. Enforce execution freshness: r > C ---- */
    if (requested_round <= round_counter)
        return TEE_ERROR_SECURITY;

    /* ---- 2. Unseal DH secrets and re-derive PRG keys ---- */
    uint8_t local_secrets[CHRONOS_MAX_PEERS][CHRONOS_KEY_SIZE];
    res = unseal_seeds((uint8_t *)local_secrets, num_peers * CHRONOS_KEY_SIZE);
    if (res != TEE_SUCCESS) return res;

    uint8_t local_prg_keys[CHRONOS_MAX_PEERS][16];
    for (uint32_t j = 0; j < num_peers; j++) {
        res = hkdf_sha256(local_secrets[j], CHRONOS_KEY_SIZE,
                          "chronos-prg", 11, local_prg_keys[j], 16);
        if (res != TEE_SUCCESS) return res;
    }

    /* ---- 3. Zero the output mask ---- */
    TEE_MemFill(mask_buffer, 0, dimension_D * sizeof(uint32_t));

    /* ---- 4. Accumulate N-1 PRG streams with rejection sampling ---- */
    static const uint8_t zero_block[4] = {0};

    TEE_OperationHandle ctr_op = TEE_HANDLE_NULL;
    res = TEE_AllocateOperation(&ctr_op, TEE_ALG_AES_CTR, TEE_MODE_ENCRYPT, 128);
    if (res != TEE_SUCCESS) return res;

    for (uint32_t j = 0; j < num_peers; j++) {
        /* Build transient AES key from k^prg_{i,j} */
        TEE_ObjectHandle key_handle = TEE_HANDLE_NULL;
        TEE_Attribute attr;

        res = TEE_AllocateTransientObject(TEE_TYPE_AES, 128, &key_handle);
        if (res != TEE_SUCCESS) goto out_op;

        TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE,
                             local_prg_keys[j], 16);
        res = TEE_PopulateTransientObject(key_handle, &attr, 1);
        if (res != TEE_SUCCESS) {
            TEE_FreeTransientObject(key_handle);
            goto out_op;
        }

        TEE_SetOperationKey(ctr_op, key_handle);

        /* IV = round index r (little-endian in first 4 bytes, rest zero) */
        uint8_t iv[16] = {0};
        memcpy(iv, &requested_round, sizeof(requested_round));
        TEE_CipherInit(ctr_op, iv, sizeof(iv));

        /* Generate D elements with rejection sampling into F_p */
        uint32_t generated = 0;
        while (generated < dimension_D) {
            uint32_t raw = 0;
            uint32_t out_len = sizeof(raw);

            TEE_CipherUpdate(ctr_op, zero_block, sizeof(zero_block),
                             &raw, &out_len);

            if (raw < CHRONOS_FIELD_PRIME) {
                /*
                 * Pairwise cancellation (Equation 2 of the paper):
                 *
                 *   m_i(r) = Σ_{j>i} PRG(k^prg_{i,j}, r)
                 *          - Σ_{j<i} PRG(k^prg_{j,i}, r)   mod p
                 *
                 * The peer array stores N-1 entries, skipping client_index
                 * itself.  We map array index j to the actual federation
                 * peer index: peers [0..client_index-1] keep their index,
                 * peers [client_index..N-2] map to federation index j+1.
                 */
                uint32_t peer_fed_index = (j < client_index) ? j : j + 1;

                if (peer_fed_index > client_index) {
                    /* j > i: add PRG output */
                    mask_buffer[generated] =
                        (mask_buffer[generated] + raw) % CHRONOS_FIELD_PRIME;
                } else {
                    /* j < i: subtract PRG output */
                    mask_buffer[generated] =
                        (mask_buffer[generated] - raw + CHRONOS_FIELD_PRIME)
                        % CHRONOS_FIELD_PRIME;
                }
                generated++;
            }
        }

        TEE_FreeTransientObject(key_handle);
    }

    /* ---- 5. Update counter: C ← r, flush to RPMB ---- */
    round_counter = requested_round;
    res = rpmb_write_counter(round_counter);

out_op:
    TEE_FreeOperation(ctr_op);

    /* Wipe local key material */
    TEE_MemFill(local_secrets, 0, sizeof(local_secrets));
    TEE_MemFill(local_prg_keys, 0, sizeof(local_prg_keys));

    return res;
}

/* ================================================================== */
/*  Command: DECRYPT_SHARE                                             */
/* ================================================================== */

static TEE_Result cmd_decrypt_share(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;

    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                       TEE_PARAM_TYPE_MEMREF_INPUT,
                                       TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (!sealed)
        return CHRONOS_ERR_NOT_READY;

    uint32_t peer_idx = params[0].value.a;
    if (peer_idx >= num_peers)
        return TEE_ERROR_BAD_PARAMETERS;

    if (params[1].memref.size < CHRONOS_SHARE_CT_SIZE)
        return TEE_ERROR_SHORT_BUFFER;
    if (params[2].memref.size < CHRONOS_KEY_SIZE)
        return TEE_ERROR_SHORT_BUFFER;

    const uint8_t *ct_in = (const uint8_t *)params[1].memref.buffer;

    /* Unpack: [ciphertext(32) | iv(12) | tag(16)] */
    const uint8_t *ciphertext = ct_in;
    const uint8_t *iv  = ct_in + CHRONOS_KEY_SIZE;
    const uint8_t *tag = ct_in + CHRONOS_KEY_SIZE + CHRONOS_GCM_IV_LEN;

    /* Re-derive k^enc for this peer from sealed secrets */
    uint8_t local_secret[CHRONOS_KEY_SIZE];
    uint8_t local_secrets_buf[CHRONOS_MAX_PEERS][CHRONOS_KEY_SIZE];
    res = unseal_seeds((uint8_t *)local_secrets_buf, num_peers * CHRONOS_KEY_SIZE);
    if (res != TEE_SUCCESS) return res;

    memcpy(local_secret, local_secrets_buf[peer_idx], CHRONOS_KEY_SIZE);
    TEE_MemFill(local_secrets_buf, 0, sizeof(local_secrets_buf));

    uint8_t local_enc_key[16];
    res = hkdf_sha256(local_secret, CHRONOS_KEY_SIZE,
                      "chronos-enc", 11, local_enc_key, 16);
    TEE_MemFill(local_secret, 0, CHRONOS_KEY_SIZE);
    if (res != TEE_SUCCESS) return res;

    /* Build AES-128-GCM decryption */
    TEE_ObjectHandle aes_key = TEE_HANDLE_NULL;
    TEE_Attribute attr;
    res = TEE_AllocateTransientObject(TEE_TYPE_AES, 128, &aes_key);
    if (res != TEE_SUCCESS) return res;

    TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, local_enc_key, 16);
    res = TEE_PopulateTransientObject(aes_key, &attr, 1);
    if (res != TEE_SUCCESS) {
        TEE_FreeTransientObject(aes_key);
        return res;
    }

    TEE_OperationHandle ae_op = TEE_HANDLE_NULL;
    res = TEE_AllocateOperation(&ae_op, TEE_ALG_AES_GCM, TEE_MODE_DECRYPT, 128);
    if (res != TEE_SUCCESS) {
        TEE_FreeTransientObject(aes_key);
        return res;
    }

    TEE_SetOperationKey(ae_op, aes_key);
    res = TEE_AEInit(ae_op, iv, CHRONOS_GCM_IV_LEN,
                     CHRONOS_GCM_TAG_LEN * 8, 0, CHRONOS_KEY_SIZE);
    if (res != TEE_SUCCESS) {
        TEE_FreeOperation(ae_op);
        TEE_FreeTransientObject(aes_key);
        return res;
    }

    uint32_t pt_len = CHRONOS_KEY_SIZE;
    res = TEE_AEDecryptFinal(ae_op, ciphertext, CHRONOS_KEY_SIZE,
                              params[2].memref.buffer, &pt_len,
                              (void *)tag, CHRONOS_GCM_TAG_LEN);

    TEE_FreeOperation(ae_op);
    TEE_FreeTransientObject(aes_key);
    TEE_MemFill(local_enc_key, 0, 16);

    return res;  /* TEE_ERROR_MAC_INVALID if auth fails */
}

/* ================================================================== */
/*  Command: PEEK_COUNTER                                              */
/* ================================================================== */

static TEE_Result cmd_peek_counter(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_pt = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_OUTPUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE);
    if (param_types != exp_pt)
        return TEE_ERROR_BAD_PARAMETERS;

    if (!sealed)
        return CHRONOS_ERR_NOT_READY;

    params[0].value.a = round_counter;
    return TEE_SUCCESS;
}

/* ================================================================== */
/*  Standard OP-TEE entry points                                       */
/* ================================================================== */

TEE_Result TA_CreateEntryPoint(void)
{
    /* Load persistent counter from RPMB on first invocation */
    rpmb_read_counter(&round_counter);
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    /* Wipe all secrets from RAM */
    TEE_MemFill(sk_i, 0, sizeof(sk_i));
    TEE_MemFill(dh_secrets, 0, sizeof(dh_secrets));
    TEE_MemFill(enc_keys, 0, sizeof(enc_keys));
    TEE_MemFill(prg_keys, 0, sizeof(prg_keys));
    sk_valid = 0;
    sealed = 0;
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param params[4],
                                    void **sess_ctx)
{
    (void)param_types;
    (void)params;
    (void)sess_ctx;

    /* If seeds were previously sealed (e.g. after reboot), mark as ready */
    TEE_ObjectHandle test = TEE_HANDLE_NULL;
    TEE_Result res = TEE_OpenPersistentObject(
        TEE_STORAGE_PRIVATE, CHRONOS_SEEDS_OBJ_ID,
        strlen(CHRONOS_SEEDS_OBJ_ID), TEE_DATA_FLAG_ACCESS_READ, &test);
    if (res == TEE_SUCCESS) {
        sealed = 1;
        TEE_CloseObject(test);
    }

    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess_ctx)
{
    (void)sess_ctx;
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx, uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)sess_ctx;

    switch (cmd_id) {
    case TA_CHRONOS_CMD_KEYGEN:
        return cmd_keygen(param_types, params);
    case TA_CHRONOS_CMD_COMPUTE_SEEDS:
        return cmd_compute_seeds(param_types, params);
    case TA_CHRONOS_CMD_SEAL:
        return cmd_seal(param_types, params);
    case TA_CHRONOS_CMD_GENERATE_MASK:
        return cmd_generate_mask(param_types, params);
    case TA_CHRONOS_CMD_DECRYPT_SHARE:
        return cmd_decrypt_share(param_types, params);
    case TA_CHRONOS_CMD_PEEK_COUNTER:
        return cmd_peek_counter(param_types, params);
    default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
