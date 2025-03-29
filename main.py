from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import statistics

file_name = "202503281215_SHM-1.tdms"

tdms_file = TdmsFile.read(file_name)

all_groups = tdms_file.groups()

channel1 = all_groups[0].channels()[0]

channel1_data = channel1[0:]
channel1_time = channel1.time_track()

mean = statistics.mean(channel1_data)
stddev = np.std(channel1_data)
numDataPoints = len(channel1_data)

print("original mean: {mean}")
print("original stddev: {stddev}")

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