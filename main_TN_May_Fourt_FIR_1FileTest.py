import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import resample_poly

def compute_sfm(signal_diff, eps=1e-12):
    """
    Compute the Spectral Flatness Measure (SFM) for the given difference signal.
    SFM is defined as the ratio of the geometric mean to the arithmetic mean of the power spectrum.
    """
    fft_diff = np.fft.rfft(signal_diff)
    mag_diff = np.abs(fft_diff)
    power_spectrum = mag_diff**2
    geom_mean = np.exp(np.mean(np.log(power_spectrum + eps)))
    arith_mean = np.mean(power_spectrum)
    return geom_mean / arith_mean

# ---------------------------
# Main Code
# ---------------------------

# Define your TDMS file name.
file_name = "202109220920_SHM-6.tdms"
tdms_file = TdmsFile.read(file_name)

# Select the channel "DYN1-1Y".
selected_channel = None
for group in tdms_file.groups():
    for channel in group.channels():
        if channel.name == "DYN1-1Y":
            selected_channel = channel
            break
    if selected_channel is not None:
        break

if selected_channel is None:
    raise ValueError("Channel 'DYN1-1Y' not found in the TDMS file.")

# Extract the signal data and the corresponding time track.
data = selected_channel[:]               # Raw data from the channel.
channel_time = selected_channel.time_track()

# Determine the sampling rate using the "wf_increment" property.
if "wf_increment" in selected_channel.properties:
    dt_property = float(selected_channel.properties["wf_increment"])
    Fs = 1.0 / dt_property
    print(f"Using wf_increment property: dt = {dt_property:.6f} s, Fs = {Fs:.2f} Hz")
else:
    # If not available, compute from the time track.
    dt_property = np.mean(np.diff(channel_time))
    Fs = 1.0 / dt_property
    print(f"Computed dt = {dt_property:.6f} s, Fs = {Fs:.2f} Hz")

# Set the decimation factor.
q = 2  # For example, decimate by 2.

# --- Reconstruction using Polyphase Filtering (resample_poly) ---

# Decimate the signal using polyphase filtering.
data_decimated = resample_poly(data, up=1, down=q)

# Reconstruct (upsample) the signal back to the original number of samples.
data_reconstructed = resample_poly(data_decimated, up=q, down=1)

# If necessary, trim or pad the reconstructed signal.
if len(data_reconstructed) > len(data):
    data_reconstructed = data_reconstructed[:len(data)]
elif len(data_reconstructed) < len(data):
    data_reconstructed = np.pad(data_reconstructed, (0, len(data) - len(data_reconstructed)), mode='edge')

# Compute the difference signal.
diff_signal = data - data_reconstructed

# Compute the Spectral Flatness Measure (SFM) for the difference signal.
sfm_value = compute_sfm(diff_signal)
print(f"Achieved SFM using polyphase filtering reconstruction: {sfm_value:.4f}")

# ---------------------------
# Plotting: Original vs Reconstructed
# ---------------------------
plt.figure(figsize=(12, 8))
plt.plot(channel_time, data, label="Original Data", linewidth=1)
plt.plot(channel_time, data_reconstructed, label="Reconstructed Data", linestyle="--", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Measurement Value")
plt.title(f"Channel DYN1-1Y: Original vs Reconstructed\nAchieved SFM = {sfm_value:.4f}")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Plotting: Difference Signal and its FFT
# ---------------------------
# Compute time increment for the FFT.
dt = np.mean(np.diff(channel_time))
# Compute the frequency axis for FFT of the difference signal.
f_axis_diff = np.fft.rfftfreq(len(diff_signal), d=dt)
# Compute FFT.
fft_diff = np.fft.rfft(diff_signal)
mag_diff = np.abs(fft_diff)

plt.figure(figsize=(12, 10))

# Plot difference signal in time domain.
plt.subplot(2, 1, 1)
plt.plot(channel_time, diff_signal, label="Difference Signal", color="orange", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Difference")
plt.title("Difference Signal (Original - Reconstructed)")
plt.legend()
plt.grid(True)

# Plot FFT of the difference signal.
plt.subplot(2, 1, 2)
plt.plot(f_axis_diff, mag_diff, label="FFT of Difference", color="green", linewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of the Difference Signal")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()