"""
scalability_sim.py — Active-Phase Latency Scalability Simulation (RQ3).

Mathematically models and plots the O(N) vs O(N^2) active-phase 
latency scaling for CHRONOS vs. SecAgg up to N=128 clients, 
extrapolating from the physical N=32 baseline.
"""

import numpy as np
import matplotlib.pyplot as plt

# Baseline measurements from Physical Testbed (N=32, Small CNN)
# B1 (Plaintext) = 75.0 ms
# CHRONOS = 119.0 ms
# SecAgg = 450.0 ms

# Model: Latency = Base_Network + O(N)_Server + O(N^2)_Crypto
BASE_NETWORK = 43.0   # Constant network overhead (ms)
SERVER_AGG_PER_N = 1.0 # Server linear addition per client (ms)

# CHRONOS: Base network + linear server aggregation + constant TEE overhead (44ms)
CHRONOS_TEE_OVERHEAD = 44.0

# SecAgg: O(N^2) key agreement overhead during active phase
# At N=32: Network(43) + Server(32) + Crypto = 450 -> Crypto = 375
# Crypto = k * N^2 -> k = 375 / (32^2) = 0.36621
SECAGG_K = 0.36621

def simulate_b1(n_clients):
    return BASE_NETWORK + (n_clients * SERVER_AGG_PER_N)

def simulate_chronos(n_clients):
    return BASE_NETWORK + (n_clients * SERVER_AGG_PER_N) + CHRONOS_TEE_OVERHEAD

def simulate_secagg(n_clients):
    crypto_overhead = SECAGG_K * (n_clients ** 2)
    return BASE_NETWORK + (n_clients * SERVER_AGG_PER_N) + crypto_overhead

def main():
    # We plot the specific evaluation points
    n_values = np.array([32, 64, 100, 128])
    
    b1_latencies = [simulate_b1(n) for n in n_values]
    chronos_latencies = [simulate_chronos(n) for n in n_values]
    secagg_latencies = [simulate_secagg(n) for n in n_values]

    # Print LaTeX coordinates for direct copy-pasting
    print("=== LaTeX Coordinates ===")
    print("% B1 (Plaintext)")
    for n, l in zip(n_values, b1_latencies):
        print(f"    ({n}, {l:.1f})")
        
    print("\n% CHRONOS")
    for n, l in zip(n_values, chronos_latencies):
        print(f"    ({n}, {l:.1f})")
        
    print("\n% SecAgg")
    for n, l in zip(n_values, secagg_latencies):
        print(f"    ({n}, {l:.1f})")
    print("=========================\n")

    # Generate Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, b1_latencies, marker='s', color='blue', linestyle='--', label='B1 (Plaintext)')
    plt.plot(n_values, chronos_latencies, marker='o', color='green', label='CHRONOS (O(N) Active-Phase)')
    plt.plot(n_values, secagg_latencies, marker='x', color='red', label='SecAgg (O(N^2) Active-Phase)')
    
    plt.axvline(x=32, color='gray', linestyle=':', label='Physical Limit (N=32)')
    
    plt.title('Simulated Scalability: Active-Phase Latency vs. Cohort Size (N)')
    plt.xlabel('Number of Clients (N)')
    plt.ylabel('Per-Round Active Latency (ms)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('scalability_simulation.png', dpi=300)
    print("Simulation plot saved to scalability_simulation.png")

if __name__ == "__main__":
    main()