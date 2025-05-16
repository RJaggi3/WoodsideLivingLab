import os
from glob import glob
from file_operations.load import load_all_channels

compressed_folder = "compressed"

# Get all zip files in the folder
zip_files = glob(os.path.join(compressed_folder, "*.tdms"))

# Iterate and print file names

shm6ZipFile = load_all_channels(zip_files[0])
shm62021 = load_all_channels("tdms_files\\202109220920_SHM-6.tdms")


for x in range(len(shm62021)):
    shm6Zip_channel = shm6ZipFile[x]
    shm62021_channel = shm62021[x]
    data_zip = shm6Zip_channel['data']
    data_orig = shm62021_channel['data']

    for y in range(len(data_orig)):
        print(f"Compressed {shm6Zip_channel['channel_name']}: {data_zip[y]}")
        print(f"Original {shm62021_channel['channel_name']}: {data_orig[y]}")

        sub = data_zip[y] - data_orig[y]
        print(f"{sub}")

        if sub > 0 or sub < 0:
            print("FUCK")