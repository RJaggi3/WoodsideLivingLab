import os
import zipfile
from glob import glob
from file_operations.load import load_all_channels

tdms_folder = "tdms_files"
zipped_folder = "zipped"
unzipped_folder = "unzipped"

os.makedirs(zipped_folder, exist_ok=True)
os.makedirs(unzipped_folder, exist_ok=True)

# Step 1: Zip each TDMS file and report compression ratio
tdms_files = glob(os.path.join(tdms_folder, "*.tdms"))
compression_ratios = {}

for tdms_path in tdms_files:
    filename = os.path.basename(tdms_path)
    zip_path = os.path.join(zipped_folder, filename + ".zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tdms_path, arcname=filename)

    original_size = os.path.getsize(tdms_path)
    zipped_size = os.path.getsize(zip_path)
    ratio = zipped_size / original_size if original_size > 0 else 0

    compression_ratios[filename] = ratio
    print(f"{filename} compressed to {ratio:.2%} of original size.")

# Step 2: Unzip each and compare data
summary = {}

for tdms_path in tdms_files:
    filename = os.path.basename(tdms_path)
    zip_path = os.path.join(zipped_folder, filename + ".zip")
    extract_path = os.path.join(unzipped_folder, filename)

    # Extract the zipped file
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extract(filename, path=unzipped_folder)

    try:
        original_channels = load_all_channels(tdms_path)
        unzipped_channels = load_all_channels(extract_path)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        continue

    mismatch_sum = 0
    mismatch_count = 0

    for ch_idx in range(len(original_channels)):
        orig_channel = original_channels[ch_idx]
        unz_channel = unzipped_channels[ch_idx]

        data_orig = orig_channel['data']
        data_unz = unz_channel['data']

        if len(data_orig) != len(data_unz):
            print(f"Length mismatch in channel {orig_channel['channel_name']}: {len(data_orig)} vs {len(data_unz)}")
            continue

        for i in range(len(data_orig)):
            diff = data_orig[i] - data_unz[i]
            if diff != 0:
                mismatch_sum += abs(diff)
                mismatch_count += 1

    avg_mismatch = mismatch_sum / mismatch_count if mismatch_count > 0 else 0.0
    summary[filename] = avg_mismatch

# Step 3: Final Summary
print("\n=== COMPRESSION & MISMATCH SUMMARY ===")
for filename in summary:
    print(f"{filename}: Compression ratio = {compression_ratios[filename]:.2%}, Average mismatch = {summary[filename]}")