def extract_sensor_channels(all_channels, sensors):

    extracted = {}

    for sensor in sensors.keys():
        extracted[sensor] = {"X": None, "Y": None}

        for channel in all_channels:
            channel_name = channel["channel_name"]
            if sensor in channel_name:
                if channel_name.endswith("X"):
                    extracted[sensor]["X"] = channel["data"]
                elif channel_name.endswith("Y"):
                    extracted[sensor]["Y"] = channel["data"]
    
    return extracted

