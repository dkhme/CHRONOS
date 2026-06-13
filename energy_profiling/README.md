# INA226 Energy Profiling Rig

This directory contains the firmware and logging scripts used to profile the Rock Pi 4's energy consumption during the active phase of the CHRONOS protocol.

## Contents
- `ina226_logger.ino`: Arduino firmware for the ESP32 bare-metal logger. Configures the INA226 over I2C at 400kHz and samples at ~900Hz.
- `sync_integration.py`: Python script to parse the serial output, identify the GPIO synchronization boundaries (active-phase boundary), and perform trapezoidal integration ($\int V \times I \, dt$) to calculate the active-phase energy footprint.

## Methodology
The 5V input rail of the Rock Pi 4 is instrumented with a 10m$\Omega$ shunt resistor. The INA226 measures the voltage drop across the shunt. GPIO 4 on the Rock Pi 4 is asserted HIGH at the start of the active phase and LOW at completion. The ESP32 captures these edges to isolate the exact energy overhead of the active-phase boundary.