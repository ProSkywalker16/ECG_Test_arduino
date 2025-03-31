import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# File paths
input_path = r"d:/ECG_Test_arduino/data/processed/ecg_cleaned.csv"
output_path = r"d:/ECG_Test_arduino/data/processed/ecg_segments.csv"

# Sampling frequency (Hz)
SAMPLING_RATE = 250

# Ensure input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load cleaned ECG data
df = pd.read_csv(input_path)
print(f"Loaded data with {len(df)} rows")

# Ensure required column exists
if "ECG_Clean" not in df.columns:
    raise KeyError("Column 'ECG_Clean' not found in input CSV. Check the cleaning step.")

# Interpolate and fill missing values
df['ECG_Clean'] = df['ECG_Clean'].interpolate(method='linear', limit_direction='both')
df['ECG_Clean'] = df['ECG_Clean'].fillna(method='ffill').fillna(method='bfill')

if df['ECG_Clean'].std() == 0:
    raise ValueError("ECG_Clean data is flat (std = 0). No peaks can be detected.")

# Create a time axis
df['Time'] = np.arange(len(df)) / SAMPLING_RATE

# --------------------------------------------------
# 1. R-PEAK DETECTION (Iterative thresholding)
# --------------------------------------------------
r_peaks = []
distance_samples = int(0.15 * SAMPLING_RATE)  # ~150ms minimum distance

# Try different threshold percentiles until enough R peaks are found
for perc in [90, 85, 80, 75]:
    threshold_value = np.percentile(df['ECG_Clean'], perc)
    r_peaks, _ = find_peaks(df['ECG_Clean'], height=threshold_value, distance=distance_samples)
    if len(r_peaks) > 2:
        print(f"R-peaks detected using {perc}th percentile threshold.")
        break

if len(r_peaks) < 2:
    raise ValueError("Not enough R-peaks detected. Check signal quality or adjust thresholds.")

# --------------------------------------------------
# 2. Q and S WAVE DETECTION
#    (Search ±60ms around each R-peak)
# --------------------------------------------------
q_peaks, s_peaks = [], []
q_window = int(0.06 * SAMPLING_RATE)  # 60ms window

for r in r_peaks:
    # Q wave: search for minimum in [r - q_window, r)
    left_bound = max(0, r - q_window)
    if r > left_bound:
        q_index = left_bound + np.argmin(df['ECG_Clean'][left_bound:r])
    else:
        q_index = r
    q_peaks.append(q_index)
    
    # S wave: search for minimum in [r, r + q_window]
    right_bound = min(len(df), r + q_window)
    if right_bound > r:
        s_index = r + np.argmin(df['ECG_Clean'][r:right_bound])
    else:
        s_index = r
    s_peaks.append(s_index)

# --------------------------------------------------
# 3. T WAVE DETECTION
#    Search from a short delay after S up to a window or next R-peak (whichever comes first)
# --------------------------------------------------
t_peaks = []
t_window = int(0.45 * SAMPLING_RATE)  # up to 450ms after S
t_delay = int(0.03 * SAMPLING_RATE)     # 30ms delay after S to avoid immediate overlap

for i, s in enumerate(s_peaks):
    start_search = s + t_delay
    # If a next R exists, limit the search to just before it (with a small margin)
    if i + 1 < len(r_peaks):
        next_r = r_peaks[i + 1]
        end_search = min(len(df) - 1, next_r - t_delay, s + t_window)
    else:
        end_search = min(len(df) - 1, s + t_window)
    
    if start_search < end_search:
        segment = df['ECG_Clean'][start_search:end_search]
        if not segment.empty:
            t_index = start_search + np.argmax(segment)
            t_peaks.append(t_index)
    else:
        # If search window is invalid, skip detection
        t_peaks.append(s)

# --------------------------------------------------
# 4. P WAVE DETECTION
#    Search from a window before Q up to a short delay before Q
# --------------------------------------------------
p_peaks = []
p_window = int(0.30 * SAMPLING_RATE)  # 300ms before Q
p_delay = int(0.03 * SAMPLING_RATE)     # 30ms margin before Q

for q in q_peaks:
    end_search = max(0, q - p_delay)
    start_search = max(0, q - p_window)
    if start_search < end_search:
        segment = df['ECG_Clean'][start_search:end_search]
        if not segment.empty:
            p_index = start_search + np.argmax(segment)
            p_peaks.append(p_index)
    else:
        p_peaks.append(q)

# --------------------------------------------------
# 5. SAVE DETECTED PEAKS AND COMPUTE INTERVALS
# --------------------------------------------------
df['R_Peak'] = 0
df.loc[r_peaks, 'R_Peak'] = 1
df['Q_Peak'] = 0
df.loc[q_peaks, 'Q_Peak'] = 1
df['S_Peak'] = 0
df.loc[s_peaks, 'S_Peak'] = 1
df['T_Peak'] = 0
for t in t_peaks:
    if t < len(df):
        df.loc[t, 'T_Peak'] = 1
df['P_Peak'] = 0
for p in p_peaks:
    if p < len(df):
        df.loc[p, 'P_Peak'] = 1

# Compute intervals (store at the R-peak row)
df['P-R Interval'] = np.nan
df['Q-T Interval'] = np.nan
df['S-T Interval'] = np.nan

n_beats = min(len(r_peaks), len(p_peaks), len(q_peaks), len(s_peaks), len(t_peaks))
for i in range(n_beats):
    p_time = df['Time'].iloc[p_peaks[i]]
    r_time = df['Time'].iloc[r_peaks[i]]
    q_time = df['Time'].iloc[q_peaks[i]]
    s_time = df['Time'].iloc[s_peaks[i]]
    t_time = df['Time'].iloc[t_peaks[i]]
    
    df.loc[r_peaks[i], 'P-R Interval'] = r_time - p_time
    df.loc[r_peaks[i], 'Q-T Interval'] = t_time - q_time
    df.loc[r_peaks[i], 'S-T Interval'] = t_time - s_time

# --------------------------------------------------
# 6. SAVE RESULTS AND VISUALIZE
# --------------------------------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Segments saved to {output_path}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['ECG_Clean'], label='ECG Signal', color='blue', linewidth=1)

plt.plot(df['Time'][r_peaks], df['ECG_Clean'][r_peaks], "x", color='red', label='R Peaks', markersize=10)
plt.plot(df['Time'][q_peaks], df['ECG_Clean'][q_peaks], "o", color='purple', label='Q Peaks', markersize=7)
plt.plot(df['Time'][s_peaks], df['ECG_Clean'][s_peaks], "o", color='green', label='S Peaks', markersize=7)
plt.plot(df['Time'][t_peaks], df['ECG_Clean'][t_peaks], "^", color='magenta', label='T Peaks', markersize=8)
plt.plot(df['Time'][p_peaks], df['ECG_Clean'][p_peaks], "v", color='orange', label='P Peaks', markersize=8)

plt.xlabel('Time (s)')
plt.ylabel('ECG Amplitude (mV)')
plt.title('ECG Signal with Detected Waves (P, Q, R, S, T)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary
print("🔎 Final Detection Summary:")
print(f" - R-Peaks: {len(r_peaks)}")
print(f" - Q-Peaks: {len(q_peaks)}")
print(f" - S-Peaks: {len(s_peaks)}")
print(f" - T-Peaks: {len(t_peaks)}")
print(f" - P-Peaks: {len(p_peaks)}")
