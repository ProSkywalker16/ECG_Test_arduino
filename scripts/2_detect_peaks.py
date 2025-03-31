import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# File paths
input_path = r"d:/ECG_Test_arduino/data/processed/ecg_cleaned.csv"
output_path = r"d:/ECG_Test_arduino/data/processed/ecg_peaks.csv"

# Sampling frequency (Hz)
SAMPLING_RATE = 250

# Ensure input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load cleaned ECG data
df = pd.read_csv(input_path)
print(f"Loaded data with {len(df)} rows")

# Check that the required column exists
if "ECG_Clean" not in df.columns:
    raise KeyError("Column 'ECG_Clean' not found in input CSV. Check the cleaning step.")

# Interpolate and fill missing values
df['ECG_Clean'] = df['ECG_Clean'].interpolate(method='linear', limit_direction='both')
df['ECG_Clean'] = df['ECG_Clean'].fillna(method='ffill').fillna(method='bfill')

# Ensure data is not flat
if df['ECG_Clean'].std() == 0:
    raise ValueError("ECG_Clean data is flat (std = 0). No peaks can be detected.")

# Create time axis in seconds
time = np.arange(len(df)) / SAMPLING_RATE

# ============================================================
# Iterative Peak Detection with a fixed threshold of 340 and increased minimum distances
# ============================================================
fixed_threshold = 340

# Candidate minimum distances in samples:
# For a heart rate near 87 BPM, the RR interval should be ~0.69 sec ~ 172 samples.
# We choose candidate distances that are larger to avoid detecting extra peaks.
candidate_min_distances = [
    int(SAMPLING_RATE * 0.5),  # 125 samples (0.5 sec)
    int(SAMPLING_RATE * 0.55), # ~137 samples
    int(SAMPLING_RATE * 0.6)   # 150 samples
]

peaks = None
used_distance = None

for d in candidate_min_distances:
    peaks_candidate, properties = find_peaks(df['ECG_Clean'], height=fixed_threshold, distance=d)
    print(f"Trying minimum distance = {d} samples, detected {len(peaks_candidate)} peaks.")
    if len(peaks_candidate) > 2:
        peaks = peaks_candidate
        used_distance = d
        break

if peaks is None or len(peaks) < 2:
    raise ValueError("Not enough peaks detected. Check the signal or adjust the fixed threshold/minimum distances.")

# Mark detected peaks in the DataFrame
df['Peak'] = 0
df.loc[peaks, 'Peak'] = 1

# ============================================================
# Calculate Heart Rate (BPM)
# ============================================================
rr_intervals = np.diff(peaks) / SAMPLING_RATE  # in seconds
average_rr = np.mean(rr_intervals)
heart_rate = 60 / average_rr  # BPM

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(time, df['ECG_Clean'], label='ECG Signal', color='blue', linewidth=1)
plt.plot(time[peaks], df['ECG_Clean'][peaks], "x", color='red', label='Detected Peaks', markersize=10)
plt.axhline(y=fixed_threshold, color='green', linestyle='--', label=f'Threshold ({fixed_threshold:.2f})')
plt.xlabel('Time (s)')
plt.ylabel('ECG Amplitude')
plt.title('ECG Signal with Detected Peaks')
plt.text(0.02, 0.98, f'Heart Rate: {heart_rate:.2f} BPM', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_output_path = r"d:/ECG_Test_arduino/data/results/ecg_peaks.jpg"
os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
plt.savefig(plot_output_path, dpi=300)
plt.show()

# ============================================================
# Save Results
# ============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Peak detection results saved to {output_path}")
print(f"📊 ECG visualization saved to {plot_output_path}")
print(f"🔍 Detected {len(peaks)} peaks with Average Heart Rate: {heart_rate:.2f} BPM")
