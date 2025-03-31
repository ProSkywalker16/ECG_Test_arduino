import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Ask for user details
user_name = input("Enter your name: ").strip()
user_age = input("Enter your age: ").strip()

# File paths
input_path = r"d:/ECG_Test_arduino/data/processed/ecg_peaks.csv"
output_text_path = rf"d:/ECG_Test_arduino/data/results/{user_name}_analysis_report.txt"
output_image_path = rf"d:/ECG_Test_arduino/data/results/{user_name}_ecg_analysis.jpg"

# Ensure input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load ECG data
df = pd.read_csv(input_path)

# Fix column names
df.columns = df.columns.str.strip()

# Ensure necessary columns exist
required_cols = {"Peak", "Timestamp", "ECG Value"}
if not required_cols.issubset(df.columns):
    raise KeyError(f"Missing required columns! Available columns: {df.columns.tolist()}")

# Convert Timestamp to numeric
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors='coerce')

# If timestamps are very high (e.g. >1000), assume they are in milliseconds and convert to seconds.
if df["Timestamp"].max() > 1000:
    print("Timestamps appear to be in milliseconds. Converting to seconds.")
    df["Timestamp"] = df["Timestamp"] / 1000.0

# Extract detected peaks
peaks = df[df["Peak"] == 1]

# Ensure peaks exist
if peaks.empty:
    raise ValueError("No peaks detected! Check if peak detection worked correctly.")

# Calculate RR intervals in seconds
rr_intervals = np.diff(peaks["Timestamp"].values)
if len(rr_intervals) == 0:
    raise ValueError("RR intervals could not be computed. Check peak detection.")

# Remove outliers using the IQR method
Q1, Q3 = np.percentile(rr_intervals, [25, 75])
IQR = Q3 - Q1
valid_rr_intervals = rr_intervals[(rr_intervals >= Q1 - 1.5 * IQR) & (rr_intervals <= Q3 + 1.5 * IQR)]
if len(valid_rr_intervals) == 0:
    raise ValueError("All RR intervals were filtered out as outliers. Check data quality.")

# Compute heart rate (BPM)
avg_rr = np.mean(valid_rr_intervals)
avg_hr = 60 / avg_rr

# Compute RR variability and determine arrhythmia
rr_std = round(np.std(valid_rr_intervals), 3)
dynamic_threshold = IQR * 0.5  
arrhythmia_detected = rr_std > dynamic_threshold

# Generate ECG Report
report = f"""ECG Analysis Report for {user_name} (Age: {user_age})
------------------------------------------------
- Total Duration: {df['Timestamp'].iloc[-1]:.1f} seconds
- Average Heart Rate: {avg_hr:.2f} BPM
- RR Interval Standard Deviation: {rr_std} s
- Variability Threshold: {dynamic_threshold:.3f} s
- Possible Arrhythmia Detected: {'Yes' if arrhythmia_detected else 'No'}
"""

# Ensure output directory exists
os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

# Save report as a text file
with open(output_text_path, "w") as f:
    f.write(report)

print(report)
print(f"Report saved to {output_text_path}")

# ------------------ PLOTTING ------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Timestamp"], df["ECG Value"], label="ECG Signal", color="blue", alpha=0.6)
plt.scatter(peaks["Timestamp"], peaks["ECG Value"], color="red", label="Detected Peaks", zorder=3)

# Highlight anomalies if arrhythmia is detected (here we plot a horizontal line at the mean)
if arrhythmia_detected:
    plt.axhline(y=np.mean(df["ECG Value"]), color="orange", linestyle="--", label="Possible Arrhythmia")

plt.title(f"ECG Analysis - {user_name} (Age: {user_age})\nAvg BPM: {avg_hr:.2f} | Arrhythmia: {'Yes' if arrhythmia_detected else 'No'}")
plt.xlabel("Time (seconds)")
plt.ylabel("ECG Value")
plt.legend()
plt.grid()

# Save plot as a JPEG file
plt.savefig(output_image_path, dpi=300)
plt.show()

print(f"ECG visualization saved as: {output_image_path}")
