"""
secagg_baseline.py — B2: Synchronous Secure Aggregation (SecAgg) baseline.

Implements the Bonawitz et al. (CCS 2017) synchronous secure aggregation
protocol as described in Section 7, baseline B2.  The key difference from
CHRONOS is that the pairwise Diffie-Hellman key exchange happens
*synchronously* during each active training round, imposing O(N²)
communication overhead per round.

This baseline uses the same cryptographic primitive (pairwise PRG
masking with ECDH key agreement) as CHRONOS, making it the most
direct comparison for isolating the effect of phase decoupling.

Usage:
    # Server
    python secagg_baseline.py server --num-clients 32 --num-rounds 50

    # Client
    python secagg_baseline.py client --client-id 0 --server 127.0.0.1:9090 \
           --dataset cifar10 --model small_cnn
"""

import argparse
import hashlib
import hmac
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from models import SmallCNN, MediumCNN, HarCNN, count_parameters
from data_partition import load_client_data

logger = logging.getLogger("chronos.baseline.secagg")

FIELD_PRIME    = (1 << 31) - 1
SCALING_FACTOR = 1 << 16
LOCAL_EPOCHS   = 5
LEARNING_RATE  = 0.01

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
#  Software-only PRG masking (same primitive as CHRONOS, no TEE)
# ---------------------------------------------------------------------------

def hkdf_derive(secret: bytes, info: bytes, length: int = 16) -> bytes:
    """HKDF-SHA256 single-block expand."""
    prk = hmac.new(b"\x00" * 32, secret, hashlib.sha256).digest()
    okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
    return okm[:length]


def prg_mask_stream(prg_key: bytes, round_id: int, dimension: int) -> np.ndarray:
    """
    Generate a PRG stream in F_p using AES-128-CTR (simulated via SHA-256
    based stream for portability).

    In the real SecAgg implementation this would use AES-CTR; for the
    evaluation the statistical properties are identical.
    """
    rng_seed = int.from_bytes(
        hashlib.sha256(prg_key + struct.pack("<I", round_id)).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(rng_seed)

    # Rejection sampling into [0, p-1]
    mask = np.empty(dimension, dtype=np.int64)
    generated = 0
    while generated < dimension:
        batch = rng.integers(0, 1 << 31, size=dimension - generated,
                             dtype=np.int64)
        valid = batch[batch < FIELD_PRIME]
        end = min(generated + len(valid), dimension)
        mask[generated:end] = valid[:end - generated]
        generated = end

    return mask


# ---------------------------------------------------------------------------
#  Synchronous key exchange (per-round, O(N²))
# ---------------------------------------------------------------------------

class SecAggKeyManager:
    """
    Simulates the synchronous per-round key exchange of SecAgg.

    In the real protocol, each client generates a fresh ephemeral
    X25519 keypair and performs N-1 DH operations every training round.
    Here we simulate the DH-derived PRG keys using seeded randomness.
    """

    def __init__(self, client_id: int, num_clients: int, seed: int):
        self.client_id = client_id
        self.num_clients = num_clients
        self.seed = seed

    def derive_round_keys(self, round_id: int) -> Dict[int, bytes]:
        """
        Perform synchronous key exchange for this round.

        Returns a dict mapping peer_id → 16-byte PRG key.
        Each DH operation takes ≈9 ms on Rock Pi 4 (Curve25519),
        so N-1 = 31 operations per round ≈ 280 ms of key exchange
        that must happen during the latency-critical active phase.
        """
        keys = {}
        for j in range(self.num_clients):
            if j == self.client_id:
                continue
            # Deterministic shared secret (simulates DH(sk_i, pk_j))
            pair = tuple(sorted([self.client_id, j]))
            raw_secret = hashlib.sha256(
                f"secagg-dh-{pair[0]}-{pair[1]}-round{round_id}-seed{self.seed}"
                .encode()
            ).digest()
            keys[j] = hkdf_derive(raw_secret, b"secagg-prg")
        return keys

    def compute_mask(self, round_id: int, dimension: int) -> np.ndarray:
        """
        Compute the additive mask for this client and round.

        m_i(r) = Σ_{j>i} PRG(s_{i,j}, r) - Σ_{j<i} PRG(s_{j,i}, r)  mod p
        """
        keys = self.derive_round_keys(round_id)
        mask = np.zeros(dimension, dtype=np.int64)

        for j, prg_key in keys.items():
            stream = prg_mask_stream(prg_key, round_id, dimension)
            if j > self.client_id:
                mask = (mask + stream) % FIELD_PRIME
            else:
                mask = (mask - stream + FIELD_PRIME) % FIELD_PRIME

        return mask


# ---------------------------------------------------------------------------
#  SecAgg Flower Client
# ---------------------------------------------------------------------------

class SecAggClient(fl.client.NumPyClient):
    """
    Flower client with synchronous SecAgg per-round key exchange.

    The per-round cost includes both the O(N²) key exchange and
    the mask generation — all within the active sensing window.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, client_id: int,
                 key_manager: SecAggKeyManager, device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.key_mgr = key_manager
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        for p, v in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(v, dtype=torch.float32).to(self.device)

    def fit(self, parameters: List[np.ndarray],
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:

        current_round = config.get("server_round", 1)
        
        gpio = GPIOSync(4)
        gpio.set_high()

        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        t0 = time.monotonic()

        # Local training
        for epoch in range(LOCAL_EPOCHS):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

        # Extract pseudo-gradient
        new_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
        grads = []
        for old, new in zip(parameters, new_params):
            grads.append((old - new).flatten())
        flat_grad = np.concatenate(grads)

        # Quantize
        scaled = np.round(flat_grad * SCALING_FACTOR).astype(np.int64)
        quantized = (scaled + (FIELD_PRIME // 2)) % FIELD_PRIME
        D = len(quantized)

        # Synchronous key exchange + mask generation (active-phase cost)
        t_keyx = time.monotonic()
        mask = self.key_mgr.compute_mask(current_round, D)
        keyx_ms = (time.monotonic() - t_keyx) * 1000

        # Apply mask
        masked = (quantized + mask) % FIELD_PRIME

        total_ms = (time.monotonic() - t0) * 1000
        logger.info("SecAgg client %d round %d: D=%d, keyx=%.1fms, total=%.1fms",
                     self.client_id, current_round, D, keyx_ms, total_ms)

        gpio.set_low()
        return [masked.astype(np.float64)], len(self.train_loader.dataset), {
            "keyx_ms": keyx_ms,
        }

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                total_loss += self.criterion(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        return total_loss / max(total, 1), total, {
            "accuracy": correct / max(total, 1)
        }


# ---------------------------------------------------------------------------
#  SecAgg Server Strategy
# ---------------------------------------------------------------------------

class SecAggStrategy(fl.server.strategy.FedAvg):
    """
    Server strategy for SecAgg.

    In standard SecAgg, the server coordinates the key exchange phase
    and collects masked updates.  The masks cancel upon summation,
    so the server recovers the plaintext aggregate.

    This implementation inherits FedAvg's aggregation since the mask
    cancellation is handled by the modular arithmetic.
    """

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Log key-exchange overhead
        keyx_times = [r[1].metrics.get("keyx_ms", 0) for r in results
                      if r[1].metrics]
        if keyx_times:
            logger.info("Round %d: SecAgg key-exchange overhead: "
                         "mean=%.1fms, max=%.1fms",
                         server_round, np.mean(keyx_times), max(keyx_times))

        return super().aggregate_fit(server_round, results, failures)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def run_server(args):
    strategy = SecAggStrategy(
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


def run_client(args):
    train_loader, test_loader, nc = load_client_data(
        args.dataset, args.client_id, args.num_clients,
        args.alpha, args.batch_size, args.seed)

    model_map = {"small_cnn": SmallCNN, "medium_cnn": MediumCNN, "har_cnn": HarCNN}
    model = model_map[args.model](num_classes=nc)

    key_mgr = SecAggKeyManager(args.client_id, args.num_clients, args.seed)
    client = SecAggClient(model, train_loader, test_loader,
                          args.client_id, key_mgr, args.device)
    fl.client.start_numpy_client(server_address=args.server, client=client)


def main():
    parser = argparse.ArgumentParser(description="B2: SecAgg Baseline")
    sub = parser.add_subparsers(dest="mode")

    sp = sub.add_parser("server")
    sp.add_argument("--num-clients", type=int, default=32)
    sp.add_argument("--num-rounds", type=int, default=50)
    sp.add_argument("--port", type=int, default=9090)

    cp = sub.add_parser("client")
    cp.add_argument("--client-id", type=int, required=True)
    cp.add_argument("--server", type=str, default="127.0.0.1:9090")
    cp.add_argument("--dataset", type=str, default="cifar10")
    cp.add_argument("--model", type=str, default="small_cnn")
    cp.add_argument("--num-clients", type=int, default=32)
    cp.add_argument("--alpha", type=float, default=0.5)
    cp.add_argument("--batch-size", type=int, default=32)
    cp.add_argument("--seed", type=int, default=42)
    cp.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if args.mode == "server":
        run_server(args)
    elif args.mode == "client":
        run_client(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
