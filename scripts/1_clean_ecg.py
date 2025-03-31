import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# File paths
input_path = r"d:/ECG_Test_arduino/data/raw/ecg_data.csv"
output_path = r"d:/ECG_Test_arduino/data/processed/ecg_cleaned.csv"

# Sampling frequency (adjust based on your data)
SAMPLING_RATE = 250  # Hz

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def butter_lowpass_filter(data, cutoff=40, fs=SAMPLING_RATE, order=4):
    """Apply a low-pass Butterworth filter to remove high-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_r_peaks(ecg_signal, sampling_rate=SAMPLING_RATE):
    """Detect R-peaks and compute heart rate."""
    peaks, _ = find_peaks(ecg_signal, height=np.percentile(ecg_signal, 95), distance=0.6 * sampling_rate)
    rr_intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
    heart_rates = 60 / rr_intervals  # Convert RR intervals to BPM

    avg_hr = np.mean(heart_rates) if heart_rates.size > 0 else None
    return peaks, heart_rates, avg_hr

def visualize_cleaning(data, avg_hr):
    """Plot raw and cleaned ECG signals separately and display average BPM."""
    plt.figure(figsize=(12, 6))

    # First subplot: Raw ECG
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data["ECG Value"], label="Raw ECG", color="red", alpha=0.7)
    plt.title("Raw ECG Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid(True)

    # Second subplot: Cleaned ECG
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data["ECG_Clean"], label="Cleaned ECG", color="blue", linewidth=1.2)
    plt.title("Cleaned ECG Signal (Filtered & Interpolated)")
    plt.xlabel("Time (samples)")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.grid(True)

    # Display average heart rate on the cleaned ECG plot
    if avg_hr:
        plt.text(0.05 * len(data), max(data["ECG_Clean"]) * 0.9, f"Avg BPM: {avg_hr:.2f}",
                 fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.6))

    plt.tight_layout()
    plt.show()

def clean_ecg(data, duration=60):
    """Clean ECG signal and compute heart rate."""
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    # Limit to first minute
    max_rows = int(SAMPLING_RATE * duration)
    data = data.iloc[:max_rows].copy()

    # Remove lead-off artifacts
    mask = (data['LO+ Status'] == 1) | (data['LO- Status'] == 1)
    data.loc[mask, 'ECG Value'] = np.nan

    # Interpolate missing values
    data['ECG_Clean'] = data['ECG Value'].interpolate(method='linear', limit_direction='both')
    data['ECG_Clean'] = data['ECG_Clean'].fillna(method='ffill').fillna(method='bfill')

    # Apply low-pass filter
    data['ECG_Clean'] = butter_lowpass_filter(data['ECG_Clean'])

    # Compute heart rate
    peaks, heart_rates, avg_hr = detect_r_peaks(data['ECG_Clean'])

    return data, heart_rates, avg_hr

def main():
    """Main function to load, clean, analyze, and visualize ECG data."""
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)

        required_columns = {'ECG Value', 'LO+ Status', 'LO- Status'}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        cleaned_df, heart_rates, avg_hr = clean_ecg(df)

        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

        visualize_cleaning(cleaned_df, avg_hr)

        # Display BPM values
        print("\nInstantaneous Heart Rate (BPM):", np.round(heart_rates, 2).tolist())
        print(f"\nAverage Heart Rate: {avg_hr:.2f} BPM" if avg_hr else "Heart rate could not be computed.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
