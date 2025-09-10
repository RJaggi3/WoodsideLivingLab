from load import load_all_channels
from extract_sensor_channels import extract_sensor_channels
from sensor_dictionary import sensor_dictionary
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from SSI_input import build_input_matrix
import PyOMA as OMA
from decimate import scipy_decimate

file = "oma_sem2/extract_data/202503281215_SHM-6.tdms"
shm6Channels = load_all_channels(file)
print(shm6Channels)
decimated_channels = scipy_decimate(shm6Channels, q=2, ftype='iir', zero_phase=True)

print(f"Total points of decimated: {len(decimated_channels[0]['data'])}")

sensors = sensor_dictionary()
extracted_channels = extract_sensor_channels(decimated_channels, sensors)
print(f"Total points of extracted: {len(extracted_channels['DYN1-2']['X'])}")


data_2d = build_input_matrix(extracted_channels)

print(f"Channels length: {len(data_2d[0])}")

data_dt = signal.detrend(data_2d, axis=0)

channel1_time = shm6Channels[0]['time']
time_diffs = np.diff(channel1_time)  

fs = 1 / np.mean(time_diffs)*0.5 
print(fs)

#fig, results = OMA.SSIcovStaDiag(data_dt, fs, br=15)
fig, results = OMA.FDDsvp(data_dt, fs)
FreQ = [13.8, 21.64, 49.1, 73.2]
print(results)
plt.show()
#results_SSIModEx = OMA.SSIModEX(FreQ, results)

#print(results_SSIModEx)