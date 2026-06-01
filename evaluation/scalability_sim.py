"""
scalability_sim.py — Active-Phase Latency Scalability Simulation (RQ3).

Mathematically models and plots the O(N) vs O(N^2) active-phase 
latency scaling for CHRONOS vs. SecAgg up to N=128 clients, 
extrapolating from the physical N=32 baseline on GCP c2-standard-8.
"""

import numpy as np
import matplotlib.pyplot as plt

# Baseline measurements from Table 4 (N=20, Small CNN, CIFAR-10)
# All times in milliseconds (ms)

CHRONOS_BASE_COMPUTE = 1.1      # TEE Mask Generation
CHRONOS_BASE_NETWORK = 71.0     # Model Download + Upload (200 KB)
CHRONOS_SERVER_AGG = 0.5        # Modular addition

SECAGG_BASE_COMPUTE = 36.0      # Software PRG + Masking
SECAGG_BASE_NETWORK = 71.0      # Model Download + Upload (200 KB)
SECAGG_SYNC_KEY_EXCH = 35.0     # Synchronous Diffie-Hellman overhead (N=20)
SECAGG_SERVER_AGG = 10.0        # De-masking and aggregation

def simulate_chronos(n_clients):
    """
    CHRONOS active phase is O(N) at the server, but O(1) for the client.
    The bottleneck is the client's network upload and TEE generation.
    Server aggregation scales linearly but is extremely fast (<1ms).
    """
    tee_time = CHRONOS_BASE_COMPUTE
    network_time = CHRONOS_BASE_NETWORK + (n_clients * 0.2) # Slight network contention
    server_time = CHRONOS_SERVER_AGG * (n_clients / 20.0)
    
    return tee_time + network_time + server_time

def simulate_secagg(n_clients):
    """
    SecAgg active phase requires synchronous O(N^2) secret sharing.
    """
    compute_time = SECAGG_BASE_COMPUTE
    network_time = SECAGG_BASE_NETWORK + (n_clients * 0.2)
    
    # O(N^2) scaling factor based on the N=20 baseline (35ms)
    # Factor = 35 / (20^2) = 0.0875
    key_exchange_time = 0.0875 * (n_clients ** 2)
    
    server_time = SECAGG_SERVER_AGG * (n_clients / 20.0)
    
    return compute_time + network_time + key_exchange_time + server_time

def main():
    n_values = np.arange(5, 70, 5)
    chronos_latencies = [simulate_chronos(n) for n in n_values]
    secagg_latencies = [simulate_secagg(n) for n in n_values]

    plt.figure(figsize=(8, 5))
    
    plt.plot(n_values, chronos_latencies, marker='o', color='green', 
             label='CHRONOS (O(N) Active-Phase)')
    plt.plot(n_values, secagg_latencies, marker='x', color='red', 
             label='SecAgg (O(N^2) Active-Phase)')
    
    # Mark the physical vs simulated boundary
    plt.axvline(x=20, color='gray', linestyle='--', label='Physical Limit (N=20)')
    
    plt.title('Simulated Scalability: Active-Phase Latency vs. Cohort Size (N)')
    plt.xlabel('Number of Clients (N)')
    plt.ylabel('Per-Round Active Latency (ms)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('scalability_simulation.png', dpi=300)
    print("Simulation complete. Plot saved to scalability_simulation.png")
    
    print("\nExtrapolated values at N=128:")
    print(f"CHRONOS: {simulate_chronos(128):.1f} ms")
    print(f"SecAgg:  {simulate_secagg(128):.1f} ms")

if __name__ == "__main__":
    main()
