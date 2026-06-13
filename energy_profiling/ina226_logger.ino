#include <Wire.h>

/* 
 * INA226 High-Speed Logger for CHRONOS Energy Profiling
 * 
 * Target: ESP32 or similar fast microcontroller.
 * Polls the INA226 power monitor at ~900Hz via 400kHz I2C.
 * Uses a hardware interrupt on SYNC_PIN to precisely bound the active 
 * active-phase boundary (triggered by Rock Pi 4 GPIO).
 */

#define INA226_ADDRESS 0x40
#define INA226_REG_CONFIG 0x00
#define INA226_REG_SHUNTVOLTAGE 0x01
#define INA226_REG_BUSVOLTAGE 0x02

// Sync pin connected to Rock Pi 4 GPIO for active-phase boundary tracking
#define SYNC_PIN 15

volatile bool is_active = false;
volatile uint32_t active_start_time = 0;

// Hardware interrupt limits jitter compared to polling
void IRAM_ATTR onSyncToggle() {
    is_active = digitalRead(SYNC_PIN);
    if (is_active) {
        // Record precise start boundary
        active_start_time = micros();
    }
}

void setup() {
    // 2M baud rate required to prevent serial buffer blocking at 900Hz
    Serial.begin(2000000); 
    
    pinMode(SYNC_PIN, INPUT_PULLDOWN);
    attachInterrupt(digitalPinToInterrupt(SYNC_PIN), onSyncToggle, CHANGE);

    Wire.begin();
    Wire.setClock(400000); // 400kHz Fast I2C mode

    /* 
     * Configure INA226 for fast conversion:
     * Avg = 1, VBUS CT = 204us, VSH CT = 140us, Mode = Shunt and Bus Continuous
     * Register map: 0100 (Avg=1) | 010 (204us) | 001 (140us) | 111 (Cont) -> 0x4247
     */
    Wire.beginTransmission(INA226_ADDRESS);
    Wire.write(INA226_REG_CONFIG);
    Wire.write(0x42);
    Wire.write(0x47);
    Wire.endTransmission();
    
    // Print CSV Header
    Serial.println("timestamp_us,is_active,bus_v,shunt_uv");
}

void loop() {
    // Only log aggressively during the active aggregation phase
    if (!is_active) {
        delay(1);
        return;
    }

    uint32_t t_micros = micros() - active_start_time;

    // Read Bus Voltage (16-bit)
    Wire.beginTransmission(INA226_ADDRESS);
    Wire.write(INA226_REG_BUSVOLTAGE);
    Wire.endTransmission();
    Wire.requestFrom(INA226_ADDRESS, 2);
    uint16_t bus_raw = (Wire.read() << 8) | Wire.read();

    // Read Shunt Voltage (16-bit, signed)
    Wire.beginTransmission(INA226_ADDRESS);
    Wire.write(INA226_REG_SHUNTVOLTAGE);
    Wire.endTransmission();
    Wire.requestFrom(INA226_ADDRESS, 2);
    int16_t shunt_raw = (Wire.read() << 8) | Wire.read();

    // LSB Scales per INA226 Datasheet
    // Bus voltage LSB = 1.25mV
    float bus_v = bus_raw * 0.00125f;
    
    // Shunt voltage LSB = 2.5uV
    float shunt_uv = shunt_raw * 2.5f;

    // Output to serial for Python integration
    Serial.print(t_micros);
    Serial.print(",");
    Serial.print(is_active);
    Serial.print(",");
    Serial.print(bus_v, 4);
    Serial.print(",");
    Serial.println(shunt_uv, 1);
    
    // Micro-delay to achieve ~900Hz target sampling rate without I2C collision
    delayMicroseconds(500); 
}