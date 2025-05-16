import numpy as np
from scipy.signal import decimate
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
import os


src_path = "tdms_files/202305282340_SHM-6.tdms"
filename = os.path.basename(src_path)
dst_folder = "decimated_files"
dst_path = os.path.join(dst_folder, filename)


os.makedirs(dst_folder, exist_ok=True)

decimation_factor = 2


original_file = TdmsFile.read(src_path)

root_object = RootObject(properties=original_file.properties)
group_objects = []
channel_objects = []

for group in original_file.groups():
    group_name = group.name
    group_objects.append(GroupObject(group_name, properties=group.properties))

    for channel in group.channels():
        original_data = channel[:]
        mean_val = np.mean(original_data)
        demeaned = original_data - mean_val
        decimated_data = decimate(demeaned, decimation_factor, ftype='iir', zero_phase=True)
        recentered_data = decimated_data + mean_val

        ch_obj = ChannelObject(
            group=group_name,
            name=channel.name,
            data=recentered_data,
            properties=channel.properties.copy()
        )
        channel_objects.append(ch_obj)

with TdmsWriter(dst_path) as writer:
    writer.write_segment([root_object] + group_objects + channel_objects)

new_file = TdmsFile.read(dst_path)

for group in new_file.groups():
    print(f"Group: {group.name}")
    for channel in group.channels():
        data = channel[:]
        print(f"  Channel: {channel.name}")
        print(f"    Length: {len(data)}")
        print(f"    Mean:   {np.mean(data):.4f}")
        print(f"    Std:    {np.std(data):.4f}")
        print(f"    First 5 samples: {data[:5]}\n")
