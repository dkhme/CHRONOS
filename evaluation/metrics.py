"""
metrics.py — CHRONOS Evaluation Metrics and CI Computation.

Provides standard functions for calculating PSNR and SSIM for the
Gradient Inversion Attack (RQ6), and abstractions for parsing the
latency/energy logs (RQ1, RQ2, RQ4).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as st

# ---------------------------------------------------------------------------
#  Visual Fidelity Metrics (RQ6: Gradient Inversion Defense)
# ---------------------------------------------------------------------------

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    Expects tensors in the range [0.0, 1.0].
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    # max pixel value is 1.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM).
    A simplified 11x11 sliding window implementation.
    """
    C1 = (0.01 * 1.0)**2
    C2 = (0.03 * 1.0)**2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    return ssim_map.mean().item()

# ---------------------------------------------------------------------------
#  Confidence Interval Calculation (RQ1, RQ3)
# ---------------------------------------------------------------------------

def compute_95_ci(data_list: list) -> tuple:
    """
    Compute mean and 95% Confidence Interval for 5 random seeds.
    """
    n = len(data_list)
    mean = np.mean(data_list)
    sem = st.sem(data_list)
    # T-distribution multiplier for 95% CI with n-1 degrees of freedom
    ci = sem * st.t.ppf((1 + 0.95) / 2., n-1)
    return mean, ci

# ---------------------------------------------------------------------------
#  Log Parsers (Abstracted Hooks)
# ---------------------------------------------------------------------------

def parse_latency_logs(log_dir: str):
    """
    Parses TEE and network latency from client logs.
    """
    # Abstract implementation: In reality, reads `logs/client_*.log`
    pass

def parse_energy_measurements(csv_path: str):
    """
    Parses the Monsoon High Voltage Power Monitor logs.
    Integrates the (Current * Voltage) over the Active-Phase timestamps.
    """
    # Abstract implementation
    pass
