#!/bin/bash
# run_experiment.sh - CHRONOS Federation Orchestrator
#
# This script demonstrates how to launch a 20-client federation locally
# for 50 rounds of training across 5 random seeds to compute 95% CIs,
# as described in Section 7 of the CHRONOS paper.
#
# Note: In the physical testbed, the clients were distributed across 20
# physical Rock Pi 4 devices. This script simulates the orchestration
# locally for reproducibility.

set -e

mkdir -p logs

NUM_CLIENTS=20
ROUNDS=50
DATASET="cifar10"
MODEL="small_cnn"
SEEDS=(42 123 456 789 999)

echo "=================================================="
echo " Starting CHRONOS Evaluation: $DATASET ($MODEL)"
echo " Clients: $NUM_CLIENTS | Rounds: $ROUNDS"
echo "=================================================="

for SEED in "${SEEDS[@]}"; do
    echo "[*] Starting Run with Seed: $SEED"
    
    # 1. Start the Aggregation Server in the background
    echo "    -> Launching Server..."
    python3 server/fl_server.py \
        --num-clients $NUM_CLIENTS \
        --rounds $ROUNDS \
        --seed $SEED \
        > logs/server_seed_${SEED}.log 2>&1 &
    
    SERVER_PID=$!
    
    # Wait for the server to initialize and bind to the port
    sleep 3
    
    # 2. Start the Idle-Phase Daemons for Key Establishment
    # (In a real deployment, these run asynchronously during charging windows)
    echo "    -> Simulating Idle-Phase Key Establishment..."
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        python3 host/idle_daemon.py \
            --client-id $i \
            --server "127.0.0.1:9090" \
            --num-clients $NUM_CLIENTS \
            --threshold 13 \
            --force \
            > logs/idle_client_${i}_seed_${SEED}.log 2>&1 &
    done
    
    # Wait for all idle daemons to complete DH key exchange
    wait
    echo "       Idle-Phase Complete."

    # 3. Start the Active-Phase Training Clients
    echo "    -> Launching Active-Phase Clients..."
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        python3 host/fl_client.py \
            --client-id $i \
            --server "127.0.0.1:9090" \
            --dataset $DATASET \
            --model $MODEL \
            --seed $SEED \
            > logs/client_${i}_seed_${SEED}.log 2>&1 &
    done
    
    # Wait for the server to finish the 50 rounds
    wait $SERVER_PID
    echo "[*] Run with Seed $SEED completed."
    
    # Ensure all client processes are cleaned up
    pkill -f "host/fl_client.py" || true
    sleep 2
done

echo "=================================================="
echo " All runs complete. Check the 'logs/' directory."
echo " Use evaluation/metrics.py to parse and compute CIs."
echo "=================================================="
