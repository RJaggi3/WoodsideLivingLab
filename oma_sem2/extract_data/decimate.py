import numpy as np
from scipy.signal import decimate

def scipy_decimate(channels, q=2, ftype='iir', zero_phase=True):
    results = []

    for ch in channels:
        data = ch['data']
        time = ch['time']
        name = ch['channel_name']
        group = ch["group_name"]

        mean_original = np.mean(data)
        data_demeaned = data - mean_original
        data_decimated = decimate(data_demeaned, q, ftype=ftype, zero_phase=zero_phase)
        data_decimated += mean_original
        time_decimated = time[::q]

        results.append({
            "channel_name": name,
            "group_name": group,
            "original_data": data,
            "original_time": time,
            "data": data_decimated,
            "time": time_decimated
        })

    return results