"""
chronos_sw_baseline.py — B4: CHRONOS-SW (software-only ablation).

A software-only ablation that isolates the phase-decoupled architecture
of CHRONOS but operates entirely in the Normal World: DH keys, PRG seeds,
and the round counter are stored in the filesystem rather than the TEE.

This baseline preserves the "idle-window scheduling" latency advantage
over synchronous SecAgg (B2) but offers NO protection against a
root-compromised host OS.  It serves to quantify the incremental cost
of hardware-isolated security (TEE context switch + RPMB flush ≈ 41–99 ms)
compared to the software-only decoupled design.

Architecturally similar to Hyb-Agg (Emmaka et al., 2025).

Usage:
    # Server
    python chronos_sw_baseline.py server --num-clients 32 --num-rounds 50

    # Client
    python chronos_sw_baseline.py client --client-id 0 --server 127.0.0.1:9090 \
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

logger = logging.getLogger("chronos.baseline.sw")

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
#  Software Key Store (Normal World filesystem — no TEE)
# ---------------------------------------------------------------------------

class SoftwareKeyStore:
    """
    Software-only key management (no hardware isolation).

    Stores DH secrets and the round counter in the Normal World
    filesystem.  A root-level adversary A_SW can trivially read
    these values and de-mask all future gradients.

    This is the fundamental security gap that CHRONOS's TEE
    addresses (Section 5.1).
    """

    def __init__(self, client_id: int, num_clients: int,
                 seed: int, store_dir: str = "/tmp/chronos_sw"):
        self.client_id = client_id
        self.num_clients = num_clients
        self.seed = seed
        self.store_dir = Path(store_dir) / str(client_id)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Simulate idle-phase key establishment
        self.prg_keys = self._derive_keys()
        self.round_counter = 0

        # Persist keys to filesystem (vulnerable to A_SW)
        self._save_keys()

    def _derive_keys(self) -> Dict[int, bytes]:
        """Derive PRG keys from simulated DH exchange (done in idle phase)."""
        keys = {}
        for j in range(self.num_clients):
            if j == self.client_id:
                continue
            pair = tuple(sorted([self.client_id, j]))
            raw_secret = hashlib.sha256(
                f"chronos-sw-{pair[0]}-{pair[1]}-seed{self.seed}".encode()
            ).digest()
            prk = hmac.new(b"\x00" * 32, raw_secret, hashlib.sha256).digest()
            prg_key = hmac.new(prk, b"chronos-prg\x01", hashlib.sha256).digest()[:16]
            keys[j] = prg_key
        return keys

    def _save_keys(self):
        """Save keys to filesystem (Normal World — exposed to A_SW)."""
        key_data = {str(k): v.hex() for k, v in self.prg_keys.items()}
        (self.store_dir / "prg_keys.json").write_text(json.dumps(key_data))
        (self.store_dir / "counter.txt").write_text(str(self.round_counter))

    def generate_mask(self, round_id: int, dimension: int) -> np.ndarray:
        """
        Generate the round mask (software-only, no RPMB).

        WARNING: The round counter is stored in a plain text file.
        A root adversary can rewind it to force mask reuse,
        violating execution freshness (Section 5.2).
        """
        # Freshness check (software only — bypassable by A_SW)
        if round_id <= self.round_counter:
            raise RuntimeError(
                f"Round counter violation: {round_id} <= {self.round_counter}")

        mask = np.zeros(dimension, dtype=np.int64)

        for j, prg_key in self.prg_keys.items():
            # Generate PRG stream (same AES-CTR-equivalent as CHRONOS)
            rng_seed = int.from_bytes(
                hashlib.sha256(
                    prg_key + struct.pack("<I", round_id)
                ).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(rng_seed)

            # Rejection sampling into F_p
            stream = np.empty(dimension, dtype=np.int64)
            generated = 0
            while generated < dimension:
                batch = rng.integers(0, 1 << 31, size=dimension - generated,
                                     dtype=np.int64)
                valid = batch[batch < FIELD_PRIME]
                end = min(generated + len(valid), dimension)
                stream[generated:end] = valid[:end - generated]
                generated = end

            # Pairwise cancellation structure
            if j > self.client_id:
                mask = (mask + stream) % FIELD_PRIME
            else:
                mask = (mask - stream + FIELD_PRIME) % FIELD_PRIME

        # Update counter (filesystem — no RPMB flush)
        self.round_counter = round_id
        (self.store_dir / "counter.txt").write_text(str(self.round_counter))

        return mask


# ---------------------------------------------------------------------------
#  CHRONOS-SW Flower Client
# ---------------------------------------------------------------------------

class ChronosSWClient(fl.client.NumPyClient):
    """
    Software-only CHRONOS ablation client.

    Identical workflow to the full CHRONOS client, but the mask is
    generated in the Normal World without any TEE involvement.
    The latency saving is ≈41–99 ms (no context switch, no RPMB flush).
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, client_id: int,
                 key_store: SoftwareKeyStore, device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.key_store = key_store
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

        # Software mask generation (no TEE context switch, no RPMB)
        t_mask = time.monotonic()
        mask = self.key_store.generate_mask(current_round, D)
        mask_ms = (time.monotonic() - t_mask) * 1000

        masked = (quantized + mask) % FIELD_PRIME

        total_ms = (time.monotonic() - t0) * 1000
        logger.info("CHRONOS-SW client %d round %d: D=%d, mask=%.1fms, total=%.1fms",
                     self.client_id, current_round, D, mask_ms, total_ms)

        gpio.set_low()
        return [masked.astype(np.float64)], len(self.train_loader.dataset), {
            "mask_ms": mask_ms,
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
#  CLI
# ---------------------------------------------------------------------------

def run_server(args):
    strategy = fl.server.strategy.FedAvg(
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

    key_store = SoftwareKeyStore(args.client_id, args.num_clients, args.seed)
    client = ChronosSWClient(model, train_loader, test_loader,
                             args.client_id, key_store, args.device)
    fl.client.start_numpy_client(server_address=args.server, client=client)


def main():
    parser = argparse.ArgumentParser(description="B4: CHRONOS-SW Baseline")
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
