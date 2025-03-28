from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

file_name = "202503281215_SHM-1.tdms"

tdms_file = TdmsFile.read(file_name)

all_groups = tdms_file.groups()

channel1 = all_groups[0].channels()[0]

channel1_data = channel1[0:]
channel1_time = channel1.time_track()

plt.plot(channel1_time,channel1_data)
plt.show()

