import wfdb
import os

# Create a directory to save the dataset
save_dir = 'mitbih_data'
os.makedirs(save_dir, exist_ok=True)

# List of valid record IDs (from MIT-BIH arrhythmia database)
valid_record_ids = list(range(100, 125)) + [
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222,
    223, 228, 230, 231, 232, 233, 234
]

print(f"📦 Downloading {len(valid_record_ids)} valid records from MIT-BIH...")

for record_id in valid_record_ids:
    record_name = str(record_id)
    try:
        print(f"⬇️  Downloading record {record_name}...")
        wfdb.dl_database(
            db_dir='mitdb',
            records=[record_name],
            dl_dir=save_dir,
            keep_subdirs=False
        )
        print(f"✅ Finished: {record_name}")
    except Exception as e:
        print(f"⚠️ Failed to download {record_name}: {e}")

print("\n✅ All available valid records downloaded to ./mitbih_data")
