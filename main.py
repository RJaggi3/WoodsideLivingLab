from file_operations.load import load_all_channels
from file_operations.write import write_tdms
from decimation.decimate import scipy_decimate
from file_operations.zip import zip_file, compare_file_sizes

file_path = "tdms_files\\202109220920_SHM-6.tdms"

decimation_factors = {
    "SHM-1": 2,
    "SHM-2": 2,
    "SHM-3": 2,
    "SHM-4": 2,
    "SHM-5": 2,
    "SHM-6": 2
}

filename = file_path.split("\\")[-1]
sensor_group = filename.split("_")[-1].split(".")[0]

decimation_factor = decimation_factors[sensor_group]

channels = load_all_channels(file_path)

decimated_results = scipy_decimate(channels, decimation_factor)

output_tdms_path = "decimated_file/202503281215_SHM-1_decimated.tdms"

write_tdms(output_tdms_path, decimated_results)

zip_output_path = "decimated_file/202109220920_SHM-6_decimated.zip"
zip_output_path2 = "decimated_file/202109220920_SHM-6.zip"

zip_file(output_tdms_path, zip_output_path)
zip_file(file_path, zip_output_path2)

sizes = compare_file_sizes(file_path, output_tdms_path, zip_output_path,zip_output_path2)

print(f"Original file size:  {sizes['original_kb']:.2f} KB")
print(f"Decimated file size: {sizes['decimated_kb']:.2f} KB")
print(f"Decimated zip file size: {sizes['decimated_zipped_kb']:.2f} KB")
print(f"Original zip file size: {sizes['orginal_zipped_kb']:.2f} KB")