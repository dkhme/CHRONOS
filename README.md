# CHRONOS: Public Reference Implementation

This repository provides the reference implementation and evaluation code for **CHRONOS: Phase-Decoupled Hardware-Assisted Secure Aggregation for Federated Learning in Edge Environments**.

To satisfy data availability and reproducibility requirements, this repository contains the standalone, runnable scripts required to replicate the core architectural claims, cryptographic operations, and evaluation metrics discussed in the paper.

## Repository Structure

```text
.
├── run_experiment.sh           # Main entry point to launch the 20-client federation
├── ta/                         # OP-TEE Trusted Application (C)
│   ├── include/
│   │   └── chronos_ta.h        # TA UUID and command IDs
│   ├── chronos_ta.c            # AES-128-CTR masking, RPMB counter, HKDF, Shamir shares
│   ├── user_ta_header_defines.h# OP-TEE UUID, heap, and stack sizing
│   ├── Makefile                # OP-TEE cross-compilation makefile
│   └── sub.mk                  # OP-TEE source inclusion list
├── host/                       # Normal World Client (Python/C-wrapper)
│   ├── tee_wrapper.c           # TEE Client API (TEEC) bindings for Python
│   ├── fl_client.py            # Active-phase Flower client with F_p quantization
│   └── idle_daemon.py          # Idle-phase key orchestration and share distribution
├── server/                     # Aggregation Server
│   └── fl_server.py            # Flower server, Shamir recovery logic
├── baselines/                  # Baseline Implementations
│   ├── plaintext_baseline.py   # B1: Plaintext FedAvg
│   ├── secagg_baseline.py      # B2: Synchronous SecAgg
│   ├── smpc_baseline.py        # B3: MP-SPDZ abstraction
│   └── chronos_sw_baseline.py  # B4: Software-only CHRONOS ablation
└── evaluation/                 # Datasets, Models, and Attack Scripts
    ├── models.py               # SmallCNN, MediumCNN, and HarCNN definitions
    ├── data_partition.py       # Dirichlet non-IID splits for CIFAR-10, FEMNIST, UCI-HAR
    ├── gradient_inversion.py   # Geiping optimization-based attack (RQ6)
    ├── metrics.py              # PSNR, SSIM, and 95% CI computation utilities
    └── scalability_sim.py      # O(N) vs O(N^2) latency simulation for N > 20 (RQ3)
```

## Reproducing the Evaluation

### 1. Hardware Requirements
* **Clients:** The physical testbed uses a heterogeneous mix of Rock Pi 4 Model B (RK3399) and Orange Pi 5 (RK3588S) devices running OP-TEE 4.4.0. To execute the TA code locally, OP-TEE must be compiled with `CFG_WITH_VFP=y` and `CFG_CRYPTO_WITH_CE=y` for ARMv8 Crypto Extensions.
* **Server:** A standard x86/ARM64 Linux machine (e.g., GCP c2-standard-8) with Python 3.8+.

### 2. Running a Federation locally
The `run_experiment.sh` script automates the orchestration of a 32-client federation over 50 rounds, repeated across 5 random seeds to calculate the 95% Confidence Intervals reported in the paper.

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```
This script handles:
1. Spawning the Flower server.
2. Invoking `host/idle_daemon.py` to perform the simulated TEE Diffie-Hellman key exchange and Shamir secret sharing.
3. Launching 20 instances of `host/fl_client.py` for the active-phase training.

### 3. Running the Gradient Inversion Attack
To verify the visual privacy guarantees against the Geiping et al. attack (RQ6):
```bash
python3 evaluation/gradient_inversion.py
```
Use `evaluation/metrics.py` to compute the PSNR/SSIM of the recovered dummy image. The expected PSNR against a CHRONOS-masked gradient should be ≈8 dB (random noise).

### 4. Generating the Scalability Plot
To reproduce the $O(N)$ vs $O(N^2)$ active-phase latency chart (Figure 3) for cohort sizes up to $N=128$:
```bash
python3 evaluation/scalability_sim.py
```
This script outputs `scalability_simulation.png` and extrapolates the latency mathematically based on the physical N=32 Rock Pi / Orange Pi baseline.
