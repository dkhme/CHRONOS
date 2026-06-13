#!/usr/bin/env python3
"""
CHRONOS Trapezoidal Integration Script
Parses high-frequency serial logs from the INA226 hardware monitor and integrates
the instantaneous power over the exact TrustZone active-phase boundary.
"""

import sys
import argparse
import pandas as pd
import numpy as np

# Physical shunt resistor specification on the profiling rig
R_SHUNT_OHMS = 0.010  # 10 mOhm precision resistor

def calculate_energy(csv_path):
    print(f"Loading trace from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
        
    # Isolate data bounded strictly by the GPIO interrupt (is_active == 1)
    df_active = df[df['is_active'] == 1].copy()
    
    if df_active.empty:
        print("Error: No active phase data (is_active=1) found in log.")
        print("Ensure the Rock Pi 4 GPIO sync pin was toggled during execution.")
        return
        
    # Calculate physical current (I = V/R)
    # shunt_uv is recorded in microvolts, convert to Volts for Ohms law
    df_active['current_A'] = (df_active['shunt_uv'] * 1e-6) / R_SHUNT_OHMS
    
    # Calculate Instantaneous Power (W = V * I)
    df_active['power_W'] = df_active['bus_v'] * df_active['current_A']
    
    # Convert timestamp (microseconds) to seconds for standard Joule integration
    df_active['time_s'] = df_active['timestamp_us'] * 1e-6
    
    # Perform exact trapezoidal integration: \int P dt
    # This mitigates polling jitter inherently present in discrete sampling
    energy_joules = np.trapz(df_active['power_W'], df_active['time_s'])
    
    energy_mj = energy_joules * 1000.0
    duration_ms = (df_active['time_s'].iloc[-1] - df_active['time_s'].iloc[0]) * 1000.0
    
    print("\n" + "="*45)
    print("      INA226 ACTIVE PHASE ENERGY REPORT")
    print("="*45)
    print(f"Samples Integrated : {len(df_active)}")
    print(f"Sampling Frequency : {len(df_active) / (duration_ms / 1000.0):.1f} Hz")
    print(f"Active Duration    : {duration_ms:.2f} ms")
    print(f"Mean Power Draw    : {df_active['power_W'].mean() * 1000.0:.2f} mW")
    print(f"Peak Power Draw    : {df_active['power_W'].max() * 1000.0:.2f} mW")
    print("-"*45)
    print(f"TOTAL ACTIVE ENERGY: {energy_mj:.2f} mJ")
    print("="*45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate power from INA226 trace.")
    parser.add_argument("log_file", help="Path to the serial CSV log")
    args = parser.parse_args()
    
    calculate_energy(args.log_file)