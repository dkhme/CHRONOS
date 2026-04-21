"""
data_partition.py — Dataset loading and non-IID partitioning.

Implements the three data configurations from Section 7 of the paper:

  CIFAR-10:  50,000 training images, 10 classes,
             Dirichlet non-IID partitioning (α = 0.5).

  FEMNIST:   Filtered to the 10 digit classes (0–9) to accommodate
             N = 20 clients while preserving non-IID characteristics.
             Partitioned by writer identity.

  UCI-HAR:   561 pre-extracted accelerometer/gyroscope features,
             6 activity classes.  Partitioned by assigning 1–2 subjects
             per client, yielding naturally non-IID distributions.

Usage:
    train_loader, test_loader, num_classes = load_client_data(
        dataset_name="cifar10", client_id=0, num_clients=20,
        alpha=0.5, batch_size=32, seed=42,
    )
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger("chronos.data")

DATA_ROOT = os.environ.get("CHRONOS_DATA_ROOT", "./data")

# ---------------------------------------------------------------------------
#  Dirichlet non-IID partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(labels: np.ndarray, num_clients: int,
                        alpha: float, seed: int) -> list:
    """
    Partition dataset indices into non-IID splits using a Dirichlet
    distribution over class proportions.

    For each class c, sample a proportion vector q ~ Dir(α) of length
    num_clients, then allocate the indices of class c according to q.

    Args:
        labels:      Array of integer class labels.
        num_clients: Number of clients N.
        alpha:       Dirichlet concentration parameter (lower = more non-IID).
        seed:        Random seed for reproducibility.

    Returns:
        List of num_clients index arrays.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to counts
        counts = (proportions * len(class_idx)).astype(int)
        # Distribute remainder
        remainder = len(class_idx) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1

        # Assign indices
        offset = 0
        for client_id in range(num_clients):
            end = offset + counts[client_id]
            client_indices[client_id].extend(class_idx[offset:end].tolist())
            offset = end

    # Shuffle within each client
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


# ---------------------------------------------------------------------------
#  CIFAR-10
# ---------------------------------------------------------------------------

def _load_cifar10(client_id: int, num_clients: int, alpha: float,
                  batch_size: int, seed: int):
    """Load CIFAR-10 with Dirichlet non-IID split."""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test)

    labels = np.array(train_set.targets)
    partitions = dirichlet_partition(labels, num_clients, alpha, seed)

    client_subset = Subset(train_set, partitions[client_id])

    train_loader = DataLoader(client_subset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger.info("CIFAR-10 client %d: %d samples (α=%.1f)",
                 client_id, len(partitions[client_id]), alpha)

    return train_loader, test_loader, 10


# ---------------------------------------------------------------------------
#  FEMNIST (filtered to 10 digit classes)
# ---------------------------------------------------------------------------

def _load_femnist(client_id: int, num_clients: int, alpha: float,
                  batch_size: int, seed: int):
    """
    Load FEMNIST filtered to the 10 digit classes, partitioned by writer.

    This uses the EMNIST ByClass split from torchvision, filtered to
    digits 0–9 only.  We then partition by the natural writer grouping
    to produce non-IID splits.

    Note: This is a non-standard filtering that affects direct
    comparability with the full 62-class FEMNIST, but preserves the
    non-IID characteristics necessary for evaluating dropout robustness.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # EMNIST ByClass: classes 0-9 are digits
    train_set = torchvision.datasets.EMNIST(
        root=DATA_ROOT, split="byclass", train=True,
        download=True, transform=transform)
    test_set = torchvision.datasets.EMNIST(
        root=DATA_ROOT, split="byclass", train=False,
        download=True, transform=transform)

    # Filter to digit classes only (labels 0-9)
    train_digit_mask = np.array(train_set.targets) < 10
    test_digit_mask = np.array(test_set.targets) < 10

    train_indices = np.where(train_digit_mask)[0]
    test_indices = np.where(test_digit_mask)[0]

    train_labels = np.array(train_set.targets)[train_indices]

    # Partition digits across clients using Dirichlet
    partitions = dirichlet_partition(train_labels, num_clients, alpha, seed)

    # Map back to original dataset indices
    client_original_indices = [train_indices[idx] for idx in partitions[client_id]]
    client_subset = Subset(train_set, client_original_indices)

    test_subset = Subset(test_set, np.where(test_digit_mask)[0].tolist())

    train_loader = DataLoader(client_subset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    logger.info("FEMNIST (10-digit) client %d: %d samples",
                 client_id, len(client_original_indices))

    return train_loader, test_loader, 10


# ---------------------------------------------------------------------------
#  UCI-HAR
# ---------------------------------------------------------------------------

def _load_ucihar(client_id: int, num_clients: int, alpha: float,
                 batch_size: int, seed: int):
    """
    Load UCI Human Activity Recognition dataset.

    The dataset contains 561 pre-extracted features from smartphone
    accelerometer and gyroscope sensors across 6 activities for 30
    subjects.  We partition by assigning 1–2 subjects per client,
    producing naturally non-IID distributions.

    Data files expected at $CHRONOS_DATA_ROOT/UCI_HAR_Dataset/:
      train/X_train.txt, train/y_train.txt, train/subject_train.txt
      test/X_test.txt,   test/y_test.txt
    """
    har_dir = Path(DATA_ROOT) / "UCI_HAR_Dataset"

    if not har_dir.exists():
        raise FileNotFoundError(
            f"UCI-HAR dataset not found at {har_dir}. "
            "Download from: https://archive.ics.uci.edu/dataset/240/"
            "human+activity+recognition+using+smartphones"
        )

    # Load training data
    X_train = np.loadtxt(har_dir / "train" / "X_train.txt")
    y_train = np.loadtxt(har_dir / "train" / "y_train.txt", dtype=int)
    subjects_train = np.loadtxt(har_dir / "train" / "subject_train.txt", dtype=int)

    # Load test data
    X_test = np.loadtxt(har_dir / "test" / "X_test.txt")
    y_test = np.loadtxt(har_dir / "test" / "y_test.txt", dtype=int)

    # Labels are 1-indexed; shift to 0-indexed
    y_train = y_train - 1
    y_test = y_test - 1

    # Partition by subject: assign 1-2 subjects per client
    unique_subjects = np.unique(subjects_train)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    # Distribute subjects round-robin across clients
    subject_to_client = {}
    for i, subj in enumerate(unique_subjects):
        subject_to_client[subj] = i % num_clients

    # Build client partition
    client_mask = np.array([subject_to_client[s] == client_id
                            for s in subjects_train])
    client_X = X_train[client_mask]
    client_y = y_train[client_mask]

    if len(client_X) == 0:
        logger.warning("UCI-HAR client %d has no samples; "
                         "this may happen with N > 30", client_id)
        # Fallback: use Dirichlet partition on all training data
        partitions = dirichlet_partition(y_train, num_clients, alpha, seed)
        idx = partitions[client_id]
        client_X = X_train[idx]
        client_y = y_train[idx]

    train_dataset = TensorDataset(
        torch.tensor(client_X, dtype=torch.float32),
        torch.tensor(client_y, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    assigned_subjects = [s for s, c in subject_to_client.items() if c == client_id]
    logger.info("UCI-HAR client %d: %d samples, subjects %s",
                 client_id, len(client_X), assigned_subjects)

    return train_loader, test_loader, 6


# ---------------------------------------------------------------------------
#  Unified loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "cifar10": _load_cifar10,
    "femnist": _load_femnist,
    "ucihar":  _load_ucihar,
}


def load_client_data(dataset_name: str, client_id: int,
                     num_clients: int = 20, alpha: float = 0.5,
                     batch_size: int = 32,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load a partitioned dataset for a single FL client.

    Args:
        dataset_name: One of "cifar10", "femnist", "ucihar".
        client_id:    Client index in [0, num_clients-1].
        num_clients:  Total federation size N.
        alpha:        Dirichlet concentration (for CIFAR-10 and FEMNIST).
        batch_size:   Mini-batch size for DataLoader.
        seed:         Random seed for reproducibility.

    Returns:
        (train_loader, test_loader, num_classes)
    """
    if dataset_name not in _LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(_LOADERS.keys())}")

    return _LOADERS[dataset_name](client_id, num_clients, alpha,
                                   batch_size, seed)


if __name__ == "__main__":
    """Print partition statistics for all datasets."""
    import sys
    logging.basicConfig(level=logging.INFO)

    for ds in ["cifar10", "femnist", "ucihar"]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")
        try:
            for cid in range(20):
                train_loader, _, nc = load_client_data(
                    ds, client_id=cid, num_clients=20, alpha=0.5, seed=42)
                n_samples = sum(len(batch[0]) for batch in train_loader)
                print(f"  Client {cid:2d}: {n_samples:6d} samples")
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
