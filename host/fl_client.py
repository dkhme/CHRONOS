"""
fl_client.py — CHRONOS Active-Phase Training Wrapper.

Subclasses the Flower NumPyClient to inject hardware-assisted gradient
masking during the active federated training phase.  Implements the
seven-step fit() protocol described in Section 6.3 of the paper:

  1. PEEK_COUNTER to confirm idle-phase key establishment is complete.
  2. Train the local model for E epochs.
  3. Extract FP32 gradients.
  4. Quantize gradients to F_p via fixed-point scaling.
  5. Invoke GENERATE_MASK(D, r) in a single TEE call.
  6. Apply mask: g_tilde = (g + m(r)) mod p.
  7. Return masked gradient to the Flower communication layer.

Hardware target:  Rock Pi 4 / RK3399 running OP-TEE 4.4.0.

Usage:
    python fl_client.py --client-id 0 --server 192.168.1.100:9090 \
                        --dataset cifar10 --model small_cnn
"""

import argparse
import ctypes
import logging
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from models import SmallCNN, MediumCNN, HarCNN
from data_partition import load_client_data

logger = logging.getLogger("chronos.client")

class GPIOSync:
    """Helper to toggle Rock Pi 4 GPIO for energy measurement bracketing."""
    def __init__(self, pin: int = 4):
        self.pin = pin
        self.val_path = f"/sys/class/gpio/gpio{pin}/value"

    def set_high(self):
        try:
            with open(self.val_path, "w") as f:
                f.write("1\n")
        except IOError:
            pass

    def set_low(self):
        try:
            with open(self.val_path, "w") as f:
                f.write("0\n")
        except IOError:
            pass

# ---------------------------------------------------------------------------
#  Constants (match the paper and chronos_ta.h)
# ---------------------------------------------------------------------------

FIELD_PRIME    = (1 << 31) - 1          # p = 2^31 - 1 (Mersenne prime)
SCALING_FACTOR = 1 << 16                # S = 2^16
GRAD_CLIP_BOUND = 1000.0                # max|g|_inf safeguard (Section 6)
LOCAL_EPOCHS   = 5                      # E = 5 local epochs per round
LEARNING_RATE  = 0.01                   # η = 0.01

# TEE error sentinel
ERR_NOT_READY  = -3

# ---------------------------------------------------------------------------
#  TEE Wrapper (ctypes)
# ---------------------------------------------------------------------------

class TEEMaskGenerator:
    """Interface to the CHRONOS TA for mask generation."""

    def __init__(self, lib_path: str = "./libchronos_tee.so"):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.chronos_tee_init.restype = ctypes.c_int
        self.lib.chronos_tee_close.restype = None
        self.lib.chronos_generate_mask.restype = ctypes.c_int
        self.lib.chronos_generate_mask.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]

        rc = self.lib.chronos_tee_init()
        if rc != 0:
            raise RuntimeError(f"TEE init failed (rc={rc})")

    def generate_mask(self, round_id: int, dimension: int) -> np.ndarray:
        """
        Invoke GENERATE_MASK(D, r) — single TEE context switch.

        Returns a NumPy array of D uint32 values in [0, p-1].
        Raises RuntimeError on rollback or if idle phase is incomplete.
        """
        out = (ctypes.c_uint32 * dimension)()
        rc = self.lib.chronos_generate_mask(
            ctypes.c_uint32(round_id),
            ctypes.c_uint32(dimension),
            out,
        )
        if rc == ERR_NOT_READY:
            raise RuntimeError("TEE: idle-phase key establishment not complete")
        if rc != 0:
            raise RuntimeError(f"GENERATE_MASK failed (rc={rc})")

        return np.ctypeslib.as_array(out).copy().astype(np.int64)

    def close(self):
        self.lib.chronos_tee_close()


# ---------------------------------------------------------------------------
#  Quantization helpers
# ---------------------------------------------------------------------------

def quantize_to_field(fp32_vector: np.ndarray) -> np.ndarray:
    """
    Quantize FP32 gradients into F_p.

    Steps (Section 6.3, Step 4):
      1. Clip to the empirically verified range |g|_inf < 1000 (Section 6)
         so that N * S * max|g|_inf stays below p and cannot alias.
      2. Scale by S = 2^16 and round to nearest integer.
      3. Shift to positive domain [0, p-1] by adding p//2.

    The constant domain shift is deterministically subtracted by the
    server during de-aggregation.
    """
    clipped = np.clip(fp32_vector, -GRAD_CLIP_BOUND, GRAD_CLIP_BOUND)
    scaled = np.round(clipped * SCALING_FACTOR).astype(np.int64)
    shifted = (scaled + (FIELD_PRIME // 2)) % FIELD_PRIME
    return shifted


def dequantize_from_field(field_vector: np.ndarray) -> np.ndarray:
    """
    Inverse of quantize_to_field.  Used by the server to recover
    the FP32 aggregate after summing masked gradients modulo p.
    """
    # Undo domain shift
    signed = field_vector.astype(np.int64) - (FIELD_PRIME // 2)
    # Undo scaling
    return signed.astype(np.float32) / SCALING_FACTOR


# ---------------------------------------------------------------------------
#  CHRONOS Flower Client
# ---------------------------------------------------------------------------

class ChronosClient(fl.client.NumPyClient):
    """
    Flower client with hardware-assisted gradient masking.

    The single TEE call per round crosses the Normal-World / Secure-World
    boundary exactly once, regardless of model dimension D.
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 client_id: int, tee: TEEMaskGenerator, device: str = "cpu"):
        self.model      = model.to(device)
        self.dataloader = dataloader
        self.client_id  = client_id
        self.tee        = tee
        self.device     = device
        self.criterion  = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for p, new_val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_val, dtype=torch.float32).to(self.device)

    def fit(self, parameters: List[np.ndarray],
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Active-phase training and masking (Section 6.3).

        Steps:
          1. Confirm key establishment via PEEK_COUNTER.
          2. Update model with global parameters and train E epochs.
          3. Extract flat FP32 gradient vector.
          4. Quantize to F_p.
          5. Invoke GENERATE_MASK(D, r) — single TEE call.
          6. Compute g_tilde = (g + m(r)) mod p.
          7. Return masked gradient.
        """
        current_round = config.get("server_round", 1)
        
        # Energy bracketing: clear the previous round's active-phase window
        # (reaching fit() means this round's global model has just been
        # received). Local training below runs with the pin LOW and is
        # therefore excluded from the active-phase energy integration.
        gpio = GPIOSync(4)
        gpio.set_low()

        # Step 1: Confirm key establishment via PEEK_COUNTER
        try:
            self.tee.peek_counter()
        except RuntimeError as e:
            logger.error("Round %d: PEEK_COUNTER failed: %s", current_round, e)
            raise fl.common.DropoutException(str(e))

        # Step 2: Local training for E epochs
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        for epoch in range(LOCAL_EPOCHS):
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Step 3: Extract FP32 gradient (difference from initial params)
        new_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
        gradients = []
        for old, new in zip(parameters, new_params):
            gradients.append((old - new).flatten())  # pseudo-gradient
        flat_grad = np.concatenate(gradients)

        # Step 4: Quantize to F_p
        quantized = quantize_to_field(flat_grad)
        D = len(quantized)

        # --- Active phase begins: local gradient computation is complete and
        # cryptographic masking starts here. The pin stays HIGH through masking
        # and the subsequent network round-trip, and is cleared at the top of
        # the next round when the aggregated global model is received.
        gpio.set_high()
        t0 = time.monotonic()

        # Step 5: TEE mask generation (single context switch)
        try:
            tee_mask = self.tee.generate_mask(round_id=current_round,
                                               dimension=D)
        except RuntimeError as e:
            logger.error("Round %d: TEE mask generation failed: %s",
                          current_round, e)
            # Drop out of this round rather than send unprotected gradient
            gpio.set_low()
            raise fl.common.DropoutException(str(e))

        # Step 6: Apply mask in F_p
        masked = (quantized + tee_mask) % FIELD_PRIME

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("Round %d: D=%d, masking done in %.1f ms (active phase continues over transmit)",
                     current_round, D, elapsed_ms)

        # Step 7: Return masked gradient. The GPIO pin remains HIGH after return
        # so the energy window spans transmission and the wait for the
        # aggregated global model; it is cleared at the top of the next round.
        return [masked.astype(np.float64)], len(self.dataloader.dataset), {
            "client_id": self.client_id,
        }

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict) -> Tuple[float, int, Dict]:
        """Standard evaluation — no masking needed."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                total_loss += self.criterion(output, batch_y).item() * len(batch_y)
                preds = output.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, total, {"accuracy": accuracy}


# ---------------------------------------------------------------------------
#  Model & data selection
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "small_cnn":  lambda nc: SmallCNN(num_classes=nc),
    "medium_cnn": lambda nc: MediumCNN(num_classes=nc),
    "har_cnn":    lambda nc: HarCNN(num_classes=nc),
}


def build_client(args) -> ChronosClient:
    """Construct the Flower client from CLI arguments."""
    # Load data partition for this client
    train_loader, test_loader, num_classes = load_client_data(
        dataset_name=args.dataset,
        client_id=args.client_id,
        num_clients=args.num_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = MODEL_REGISTRY[args.model](num_classes)
    D = sum(p.numel() for p in model.parameters())
    logger.info("Model %s: D = %d parameters, %d classes",
                 args.model, D, num_classes)

    tee = TEEMaskGenerator(args.tee_lib)

    return ChronosClient(
        model=model,
        dataloader=train_loader,
        client_id=args.client_id,
        tee=tee,
        device=args.device,
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CHRONOS FL Client")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:9090")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "femnist", "ucihar"])
    parser.add_argument("--model", type=str, default="small_cnn",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--num-clients", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration for non-IID split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tee-lib", type=str, default="./libchronos_tee.so")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    client = build_client(args)
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
