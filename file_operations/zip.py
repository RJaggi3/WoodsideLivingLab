import os
import zipfile

def zip_file(input_file_path, zip_output_path):
    with zipfile.ZipFile(zip_output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file_path, arcname=os.path.basename(input_file_path))

def compare_file_sizes(original_path, decimated_path, zipped_path,zip_output_path2):
    def get_size_kb(path):
        return os.path.getsize(path) / 1024

    return {
        "original_kb": get_size_kb(original_path),
        "decimated_kb": get_size_kb(decimated_path),
        "decimated_zipped_kb": get_size_kb(zipped_path),
        "orginal_zipped_kb":get_size_kb(zip_output_path2)
    }
