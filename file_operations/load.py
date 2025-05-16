from nptdms import TdmsFile

def load_all_channels(file_path):
    tdms_file = TdmsFile.read(file_path)
    all_channels = []

    for group in tdms_file.groups():
        for channel in group.channels():
            data = channel[:]
            time = channel.time_track()
            all_channels.append({
                "group_name" : group.name,
                "channel_name" : channel.name,
                "data": data,
                "time": time
            })

    return all_channels
