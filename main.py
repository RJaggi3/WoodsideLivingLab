from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import statistics
from scipy.signal import decimate, resample, resample_poly

file_name = "202109220920_SHM-6.tdms"

tdms_file = TdmsFile.read(file_name)

all_groups = tdms_file.groups()

channel1 = all_groups[0].channels()[0]

channel1_data = channel1[0:]
channel1_time = channel1.time_track()

mean = statistics.mean(channel1_data)
stddev = np.std(channel1_data)
numDataPoints = len(channel1_data)


time_diffs = np.diff(channel1_time)  


fs = 1 / np.mean(time_diffs) 

#decimate
q = 2
mean_original = np.mean(channel1_data)
channel1_data_demeaned = channel1_data - mean_original
channel1_data_decimated = decimate(channel1_data_demeaned, q, ftype='iir', zero_phase=True)
channel1_data_decimated += mean_original  

channel1_time_decimated = channel1_time[::q]

#resample
num_resample_points = int(len(channel1_data) * fs / 50)
channel1_data_resampled = resample(channel1_data, num_resample_points)
channel1_time_resampled = np.linspace(channel1_time[0], channel1_time[-1], num_resample_points)

#resample_ploy
downsample_factor_num = 5  
downsample_factor_den = 9 
channel1_data_resample_poly = resample_poly(channel1_data_demeaned, downsample_factor_num, downsample_factor_den)
channel1_data_resample_poly += mean_original  
channel1_time_resample_poly = np.linspace(channel1_time[0], channel1_time[-1], len(channel1_data_resample_poly))
# Zoom settings
start_time = 0
end_time = 2  

# Create mask for the zoomed-in time range
mask_original = (channel1_time >= start_time) & (channel1_time <= end_time)
mask_decimated = (channel1_time_decimated >= start_time) & (channel1_time_decimated <= end_time)
mask_resampled = (channel1_time_resampled >= start_time) & (channel1_time_resampled <= end_time)
mask_resample_poly = (channel1_time_resample_poly >= start_time) & (channel1_time_resample_poly <= end_time)


plt.figure(figsize=(14, 6))

plt.plot(channel1_time[mask_original], channel1_data[mask_original],
         label="Original Signal", color='blue', linewidth=1)

plt.plot(channel1_time_decimated[mask_decimated], channel1_data_decimated[mask_decimated],
         label="Decimated Signal", color='red', linestyle='--')

plt.plot(channel1_time_resampled[mask_resampled], channel1_data_resampled[mask_resampled],
         label="Resampled", color='green', linestyle=':')

plt.plot(channel1_time_resample_poly[mask_resample_poly], channel1_data_resample_poly[mask_resample_poly],
         label="Resample Poly", color='orange', linestyle='-.')

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Comparison of Downsampling Methods (First 2 Seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting the full-length comparison
plt.figure(figsize=(14, 6))

plt.plot(channel1_time, channel1_data,
         label="Original Signal", color='blue', linewidth=1)

plt.plot(channel1_time_decimated, channel1_data_decimated,
         label="Decimated Signal", color='red', linestyle='--')

plt.plot(channel1_time_resampled, channel1_data_resampled,
         label="Resampled", color='green', linestyle=':')

plt.plot(channel1_time_resample_poly, channel1_data_resample_poly,
         label="Resample Poly", color='orange', linestyle='-.')

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Comparison of Downsampling Methods (Full Duration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()