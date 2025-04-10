from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import statistics
from scipy.signal import decimate
from scipy.signal import welch


file_name = "202503281215_SHM-1.tdms"

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

frequencies, power = welch(channel1_data, fs, nperseg=1024)

threshold = max(power) * 0.01  
max_freq_psd = max(frequencies[power > threshold])

plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, power)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Power Spectrum of the Signal")
plt.grid()
plt.show()

print(f"Estimated Maximum Frequency Component (from PSD): {max_freq_psd:.2f} Hz")


q = 13 
mean_original = np.mean(channel1_data)
channel1_data_demeaned = channel1_data - mean_original
channel1_data_decimated = decimate(channel1_data_demeaned, q, ftype='iir', zero_phase=True)
channel1_data_decimated += mean_original  

channel1_time_decimated = channel1_time[::q]

print(f"Mean (Original): {np.mean(channel1_data)}")
print(f"Mean (Decimated): {np.mean(channel1_data_decimated)}")
print(np.mean(channel1_data_decimated)/mean_original)

print(len(channel1_time_decimated))

plt.figure(figsize=(10, 5))

plt.plot(channel1_time, channel1_data, label="Original Signal", color='blue', alpha=0.6)
plt.plot(channel1_time_decimated, channel1_data_decimated, label="Decimated Signal", color='red', linestyle='--')


plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Original vs. Decimated Signal")
plt.legend()
plt.grid(True)

plt.show()



f = open("OneStandardDev.txt","a")
decimatedData = []

negEndpoint = mean - stddev
posEndpoint = mean + stddev

for x in range(len(channel1_data)):
    dataPoint = channel1_data[x]
    if dataPoint <= posEndpoint and dataPoint >= negEndpoint:
        continue
    else:
        decimatedData.append(dataPoint)

count = len(decimatedData)
newMean = statistics.mean(decimatedData)
newstddev = np.std(decimatedData)

f.write(f"Original Mean: {mean} \n")
f.write(f"Original stddev: {stddev} \n")
f.write(f"Original number of data points: {numDataPoints} \n")


f.write(f"New Mean: {newMean} \n")
f.write(f"New stddev: {newstddev} \n")
f.write(f"New number of data points: {count} \n")
f.write(f"List: {decimatedData} \n")





print(stddev)

#plt.plot(channel1_time,channel1_data)
#plt.show()

#transformation code
"""
numDataPoints = len(channel1_data)
sampleRate = 1/channel1.properties['wf_increment']



yf = fft(channel1_data)
xf = fftfreq(numDataPoints, sampleRate)

yf_shifted = np.fft.fftshift(yf)
xf_shifted = np.fft.fftshift(xf)

plt.plot(xf_shifted, 2.0/numDataPoints * np.abs(yf_shifted))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Full Frequency Spectrum from TDMS Data")
plt.grid()
plt.show()
"""