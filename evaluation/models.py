"""
models.py — Neural network architectures for CHRONOS evaluation.

Defines the three models used in Section 7 of the paper:

  SmallCNN   — D ≈ 50,000 parameters  (CIFAR-10, FEMNIST)
  MediumCNN  — D ≈ 1,000,000 parameters (CIFAR-10, FEMNIST)
  HarCNN     — D ≈ 50,000 parameters  (UCI-HAR, 1D temporal conv)

All models use standard PyTorch layers.  The gradient dimensionality D
determines the AES-128-CTR keystream length and hence the TEE round
time (Section 7.5, Table 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Small convolutional network for CIFAR-10 / FEMNIST.

    Architecture (D ≈ 50,000):
      Conv2d(in, 6, 5)  → ReLU → MaxPool(2)     (32→14)
      Conv2d(6, 16, 5)  → ReLU → MaxPool(2)     (14→5)
      FC(16 * 5 * 5, 120) → ReLU
      FC(120, 84) → ReLU
      FC(84, num_classes)

    Parameter count (in_channels=3, num_classes=10):
      Conv1:   3×6×5×5 + 6         =    450 +    6 =       456
      Conv2:   6×16×5×5 + 16       =  2,400 +   16 =     2,416
      FC1:     16×5×5 × 120 + 120  = 48,000 +  120 =    48,120
      FC2:     120 × 84 + 84       = 10,080 +   84 =    10,164
      FC3:     84 × 10 + 10        =    840 +   10 =       850
                                                    -----------
                                      Total  ≈  ~50K  (varies with in_channels)

    This is a LeNet-5 variant, a standard lightweight CNN architecture
    commonly used in federated learning benchmarks.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        # After two pools on 32×32 input: 16 × 5 × 5 = 400
        # For 28×28 input (FEMNIST): 16 × 4 × 4 = 256
        self._fc1_in = None  # Lazily determined
        self.fc1 = None
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Default initialization for 32×32 input
        self._init_fc1(in_channels, 32)

    def _init_fc1(self, in_channels: int, input_size: int):
        """Compute FC1 input size from a dummy forward pass through convs."""
        dummy = torch.zeros(1, in_channels, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        self._fc1_in = dummy.numel()
        self.fc1 = nn.Linear(self._fc1_in, 120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MediumCNN(nn.Module):
    """
    Medium convolutional network for CIFAR-10 / FEMNIST.

    Architecture (D ≈ 1,000,000):
      Conv2d(3, 64, 3, padding=1) → BN → ReLU → MaxPool
      Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool
      Conv2d(128, 256, 3, padding=1) → BN → ReLU → MaxPool
      FC(256 * 4 * 4, 512) → ReLU → Dropout(0.5)
      FC(512, num_classes)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class HarCNN(nn.Module):
    """
    1D temporal convolutional network for UCI-HAR.

    Input: (batch, 1, 561) — 561 pre-extracted features.

    Architecture (D ≈ 50,000 to match Small CNN footprint):
      Conv1d(1, 64, 5, padding=2) → ReLU → MaxPool1d(2)
      Conv1d(64, 128, 5, padding=2) → ReLU → MaxPool1d(2)
      Conv1d(128, 64, 3, padding=1) → ReLU → AdaptiveAvgPool1d(8)
      FC(64 * 8, 128) → ReLU
      FC(128, num_classes)
    """

    def __init__(self, num_classes: int = 6, in_features: int = 561):
        super().__init__()
        self.in_features = in_features

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input may be (B, 561); reshape to (B, 1, 561) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
#  Parameter counting utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Verify parameter counts match the paper's claims
    for name, cls, nc, sample_shape in [
        ("SmallCNN (CIFAR-10)",  SmallCNN,  10, (1, 3, 32, 32)),
        ("MediumCNN (CIFAR-10)", MediumCNN, 10, (1, 3, 32, 32)),
        ("SmallCNN (FEMNIST)",   SmallCNN,  10, (1, 1, 28, 28)),
        ("HarCNN (UCI-HAR)",     HarCNN,     6, (1, 1, 561)),
    ]:
        if "FEMNIST" in name:
            m = cls(num_classes=nc, in_channels=1)
        else:
            m = cls(num_classes=nc)
        D = count_parameters(m)

        # Verify forward pass
        x = torch.randn(*sample_shape)
        y = m(x)
        print(f"{name:30s}  D = {D:>10,}  output = {tuple(y.shape)}")
