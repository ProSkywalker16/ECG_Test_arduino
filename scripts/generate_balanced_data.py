import os
import wfdb
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle
import random

# Configurations
DATA_DIR = "../mitbih_data"
OUTPUT_FILE = "../scripts/preprocessed_windows_balanced.npz"
WINDOW_SIZE = 1800
ARRHYTHMIA_MAP = {
    'N': 0,  # Normal
    'A': 1,  # AFib
    'V': 2,  # PVC
    'L': 3,  # LBBB
    'R': 4,  # RBBB
    'f': 5   # Fusion
}
RECORDS = [  # All common MIT-BIH records
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Augmentation helper
def augment(signal):
    if random.random() < 0.5:
        signal += np.random.normal(0, 0.02, size=signal.shape)
    if random.random() < 0.3:
        signal *= np.random.uniform(0.9, 1.1)
    return signal

# Storage
windows_by_class = defaultdict(list)

for record_id in RECORDS:
    try:
        record = wfdb.rdrecord(os.path.join(DATA_DIR, record_id))
        annotation = wfdb.rdann(os.path.join(DATA_DIR, record_id), 'atr')
        signal = record.p_signal[:, 0]

        for i, symbol in enumerate(annotation.symbol):
            if symbol not in ARRHYTHMIA_MAP:
                continue
            label = ARRHYTHMIA_MAP[symbol]
            sample = annotation.sample[i]
            start = max(0, sample - WINDOW_SIZE // 2)
            end = start + WINDOW_SIZE
            if end <= len(signal):
                window = signal[start:end]
                window = (window - np.mean(window)) / np.std(window)
                windows_by_class[label].append(window)

    except Exception as e:
        print(f"⚠️ Skipping {record_id} due to error: {e}")

# Balancing
min_count = min(len(w) for w in windows_by_class.values())
print(f"✅ Balancing to {min_count} samples per class")

X, y = [], []
for label, windows in windows_by_class.items():
    selected = random.sample(windows, min_count)
    for win in selected:
        if label != 0:
            win = augment(win)
        X.append(win)
        y.append(label)

X, y = shuffle(np.array(X), np.array(y), random_state=42)
np.savez_compressed(OUTPUT_FILE, X=X, y=y)
print(f"✅ Saved balanced dataset to {OUTPUT_FILE}")
