from nptdms import TdmsWriter, GroupObject, ChannelObject
import numpy as np

def write_tdms(output_path, decimated_results):
    group_object = GroupObject(decimated_results[0]["group_name"])
    channel_objects = []
    for result in decimated_results:
        channel_name = result["channel_name"]
        group_name = result["group_name"]
        data = result["decimated_data"]
        time = result["decimated_time"]
        fs = 1 / np.mean(np.diff(time))

        props = {}
        props["wf_increment"] = 1 / fs
        props["wf_start_time"] = 0

        channel_obj = ChannelObject(
            group=group_name,
            channel=channel_name,
            data=data,
            properties=props,
        )
        channel_objects.append(channel_obj)

    with TdmsWriter(output_path) as writer:
        writer.write_segment([
            group_object,
            *channel_objects
        ])
