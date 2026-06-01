"""
plaintext_baseline.py — B1: Plaintext FedAvg baseline.

Standard Federated Averaging (McMahan et al., 2017) with no privacy
protection.  Serves as the latency and energy lower bound, and the
privacy upper bound (plaintext gradients are fully exposed).

This is the B1 baseline from Section 7 of the CHRONOS paper.

Usage:
    # Server
    python plaintext_baseline.py server --num-clients 32 --num-rounds 50

    # Client (run on each device)
    python plaintext_baseline.py client --client-id 0 --server 127.0.0.1:9090 \
           --dataset cifar10 --model small_cnn
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from models import SmallCNN, MediumCNN, HarCNN, count_parameters
from data_partition import load_client_data

logger = logging.getLogger("chronos.baseline.plaintext")

LOCAL_EPOCHS  = 5
LEARNING_RATE = 0.01


class PlaintextClient(fl.client.NumPyClient):
    """Standard FedAvg client — no cryptographic protection."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, client_id: int, device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        for p, v in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(v, dtype=torch.float32).to(self.device)

    def fit(self, parameters: List[np.ndarray],
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:

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

        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Client %d round %d: trained in %.1f ms",
                     self.client_id, config.get("server_round", 0), elapsed)

        return self.get_parameters({}), len(self.train_loader.dataset), {}

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


def run_server(args):
    """Launch the Flower server for plaintext FedAvg."""
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


def run_client(args):
    """Launch a single Flower client."""
    train_loader, test_loader, nc = load_client_data(
        args.dataset, args.client_id, args.num_clients,
        args.alpha, args.batch_size, args.seed)

    model_map = {"small_cnn": SmallCNN, "medium_cnn": MediumCNN, "har_cnn": HarCNN}
    model = model_map[args.model](num_classes=nc)

    client = PlaintextClient(model, train_loader, test_loader,
                             args.client_id, args.device)
    fl.client.start_numpy_client(server_address=args.server, client=client)


def main():
    parser = argparse.ArgumentParser(description="B1: Plaintext FedAvg")
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
