import os
from glob import glob
from file_operations.load import load_all_channels 
from file_operations.write import write_tdms
from decimation.decimate import scipy_decimate


input_folder = "tdms_files"
output_folder = "decimated_file"
os.makedirs(output_folder, exist_ok=True)

decimation_factors = {
    "SHM-1": 2,
    "SHM-2": 2,
    "SHM-3": 2,
    "SHM-4": 2,
    "SHM-5": 2,
    "SHM-6": 2
}

tdms_files = glob(os.path.join(input_folder, "*.tdms"))

for file_path in tdms_files:
    filename = os.path.basename(file_path)
    sensor_group = filename.split("_")[-1].split(".")[0]

    decimation_factor = decimation_factors.get(sensor_group, 2)

    channels = load_all_channels(file_path)
    decimated_results = scipy_decimate(channels, decimation_factor)


    base_name = filename.replace(".tdms", "")