import wfdb
import numpy as np
from tqdm import tqdm
import os

data_path = 'mitbih_data'
window_size = 1800  # 5 seconds at 360 Hz

# MIT-BIH valid records with annotations
valid_record_ids = list(range(100, 125)) + [
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222,
    223, 228, 230, 231, 232, 233, 234
]

# Annotation class to label map (5-class problem)
label_map = {
    'N': 0,  # Normal
    'A': 1,  # AFib
    'V': 2,  # PVC
    'B': 3,  # Bradycardia
    'S': 4   # ST Elevation (as placeholder)
}

X, y = [], []

print("🔄 Processing records and creating windowed data...")

for record_id in tqdm(valid_record_ids):
    record_name = str(record_id)
    try:
        record = wfdb.rdrecord(os.path.join(data_path, record_name))
        annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')

        signal = record.p_signal[:, 0]  # Lead II
        ann_samples = annotation.sample
        ann_symbols = annotation.symbol

        for sample, symbol in zip(ann_samples, ann_symbols):
            if symbol not in label_map:
                continue

            start = sample - window_size // 2
            end = sample + window_size // 2

            if start < 0 or end > len(signal):
                continue

            window = signal[start:end]
            X.append(window)
            y.append(label_map[symbol])

    except Exception as e:
        print(f"⚠️ Failed on record {record_name}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Final shape → X: {X.shape}, y: {y.shape}")
np.savez_compressed('processed_windows.npz', X=X, y=y)
print("📦 Saved to processed_windows.npz")
