from load import load_all_channels
from extract_sensor_channels import extract_sensor_channels
from sensor_dictionary import sensor_dictionary
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from SSI_input import build_input_matrix
from pyoma2.algorithms import FSDD, SSI, pLSCF
from pyoma2.setup import SingleSetup
from decimate import scipy_decimate

file = "oma_sem2/extract_data/202503281215_SHM-6.tdms"
shm6Channels = load_all_channels(file)


sensors = sensor_dictionary()
extracted_channels = extract_sensor_channels(shm6Channels, sensors)
print(f"Total points of extracted: {len(extracted_channels['DYN1-2']['X'])}")


data_2d = build_input_matrix(extracted_channels)
#print(data_2d)

print(f"Channels length: {len(data_2d[0])}")


data_dt = signal.detrend(data_2d, axis=0)

#estimate the sampling frequency
channel1_time = shm6Channels[0]['time']
print(f"Total points of time: {len(channel1_time)}")
time_diffs = np.diff(channel1_time)  


fs = 1 / np.mean(time_diffs)
print(f"Sampling Freg [Hz] = {fs}")


#Setup the data
Woodside_setup = SingleSetup(data_2d, fs=fs)

fig,ax = Woodside_setup.plot_data(nc=4)
plt.show()


#Plot some of interesting channels
fig_int,ax_int = Woodside_setup.plot_ch_info(ch_idx=[0])
plt.show()
'''
#filter out the electric noise at 50Hz using a bandstop filter

Woodside_setup.filter_data(Wn=0.02, btype='highpass', order=8)
fig_filt,ax_filt = Woodside_setup.plot_ch_info(ch_idx=[0])
plt.show()

#run ssi_cov
ssicov = SSI(name="SSIcov", method="cov", br=30, ordmax=50, calc_unc=True, step=2)
Woodside_setup.add_algorithms(ssicov)
Woodside_setup.run_all()    

# plot the stabilisation diagram
_, _ = ssicov.plot_stab( hide_poles=False, spectrum=True)

plt.show()
Woodside_setup.mpe_from_plot("SSIcov")

'''