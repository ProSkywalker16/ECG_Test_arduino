import csv
import random
import numpy as np

# Output CSV file
filename = "dummy_ecg_arrhythmia.csv"

# Constants
sampling_interval = 0.006944179534912109  # ~144 Hz
duration = 10  # seconds
samples = int(duration / sampling_interval)

# Reference values
adc_ref_voltage = 3.3
adc_resolution = 1024

# Define ECG-like patterns
normal_pattern = [320, 336, 475]               # NSR
tachy_pattern = [500, 520, 550]                # Fast spikes
skipped_beat_pattern = [320] * 3               # Flatline like
irregular_pattern = [336, 475, 290, 510]       # Mixed noise

# Open CSV and write header
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time_s", "raw_adc", "voltage_V"])

    current_time = 0.0
    for i in range(samples):
        # Normal for first 3 seconds
        if current_time < 3:
            pattern = normal_pattern
        # Simulate tachycardia for 3–6s
        elif current_time < 6:
            pattern = tachy_pattern
        # Skipped beats and flatlines from 6–8s
        elif current_time < 8:
            pattern = skipped_beat_pattern
        # Irregular spikes from 8–10s
        else:
            pattern = irregular_pattern

        random.shuffle(pattern)
        for adc in pattern:
            voltage = adc * adc_ref_voltage / adc_resolution
            writer.writerow([round(current_time, 8), adc, voltage])
        current_time += sampling_interval

print(f"🩺 Generated dummy ECG dataset simulating arrhythmia → {filename}")
