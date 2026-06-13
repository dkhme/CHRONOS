#!/usr/bin/env bash
# CHRONOS Network Emulation Interface (Wrapper for tc netem)
# Injects strict WAN constraints (latency, packet loss, jitter) into the cohort router.

set -e

# Default outbound interface for the gateway
IFACE="eth0"

show_help() {
    echo "Usage: $0 [start|stop|status] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start   Apply network emulation rules via tc qdisc"
    echo "  stop    Clear all network emulation rules"
    echo "  status  Show current traffic control rules"
    echo ""
    echo "Options for 'start':"
    echo "  --latency <time>    Base RTT latency (e.g., 100ms)"
    echo "  --loss <percent>    Packet loss percentage (e.g., 1%)"
    echo "  --jitter <time>     Latency variation (e.g., 10ms) (Optional)"
    echo "  --iface <name>      Target network interface (default: eth0)"
}

start_emulation() {
    local lat="0ms"
    local loss="0%"
    local jitter=""
    
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --latency) lat="$2"; shift ;;
            --loss) loss="$2"; shift ;;
            --jitter) jitter="$2"; shift ;;
            --iface) IFACE="$2"; shift ;;
            *) echo "Unknown parameter: $1"; exit 1 ;;
        esac
        shift
    done

    echo "[*] Applying WAN constraints to $IFACE..."
    echo "[*] Params -> Delay: ${lat}, Loss: ${loss}, Jitter: ${jitter:-0ms}"
    
    # Safely clear any existing root queuing disciplines
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    
    # Construct and apply new netem rules
    if [ -n "$jitter" ]; then
        tc qdisc add dev "$IFACE" root netem delay "$lat" "$jitter" loss "$loss"
    else
        tc qdisc add dev "$IFACE" root netem delay "$lat" loss "$loss"
    fi
    
    echo "[+] Network emulation active."
}

stop_emulation() {
    echo "[*] Clearing traffic control rules on $IFACE..."
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    echo "[+] Emulation stopped. Interface returned to LAN speeds."
}

status_emulation() {
    echo "Current traffic control (tc) rules for $IFACE:"
    tc qdisc show dev "$IFACE"
}

# Require root for tc
if [ "$EUID" -ne 0 ]; then
  echo "Error: This script manipulates network interfaces and must be run as root."
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
    start)
        start_emulation "$@"
        ;;
    stop)
        stop_emulation
        ;;
    status)
        status_emulation
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_help
        exit 1
        ;;
esac