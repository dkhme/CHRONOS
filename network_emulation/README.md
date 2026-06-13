# Network Emulation (netem)

This directory contains the `tc` (traffic control) and `netem` scripts used to emulate realistic WAN latencies and packet loss for the cohort evaluation in Section 7.1.

## Contents
- `emulate_wan.sh`: Bash script to configure queuing disciplines (qdiscs) on the gateway router.

## Usage
```bash
# Inject 100ms RTT with 1% packet loss
sudo ./emulate_wan.sh --latency 100ms --loss 1%

# Clear emulation
sudo ./emulate_wan.sh --clear
```