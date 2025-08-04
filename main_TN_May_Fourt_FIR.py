import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import firwin, lfilter  # We now need firwin and lfilter

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name
file_name = "202109220920_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_FIR_fractionQ.pdf")

# Read the TDMS file and collect all channels
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object so each figure becomes one PDF page
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

# Define grid: 6 rows x 6 columns
# Col 0: Time Domain (Original Data)
# Col 1: Frequency Domain (FFT of Original and Decimated)
# Col 2: Reconstructed Decimated Data (Time Domain via IFFT)
# Col 3: Difference Signal (Original - Reconstructed) in Time Domain
# Col 4: FFT of the Difference Signal (log-scale x-axis)
# Col 5: SFM for the Difference Signal (value only)
n_rows_fixed = 6
n_cols = 6

# Process channels in groups, up to 6 channels per page
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(25, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Assign each subplot axis
        ax_time     = axs[i, 0]
        ax_freq     = axs[i, 1]
        ax_recon    = axs[i, 2]
        ax_diff     = axs[i, 3]
        ax_fft_diff = axs[i, 4]
        ax_sfm      = axs[i, 5]
        
        # -----------------------------
        # TIME DOMAIN (Original Data)
        # -----------------------------
        data = channel[:]  # Raw TDMS data
        channel_time = channel.time_track()  # Corresponding time track

        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Original Data",
                     rasterized=True,
                     zorder=10)
        ax_time.set_xlabel("Time")
        y_label = channel.properties["unit_string"] if "unit_string" in channel.properties else "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # ---------------------------------------------------
        # FRACTIONAL DECIMATION USING UPSAMPLING/DOWNSAMPLING
        # ---------------------------------------------------
        # We want a decimation factor q_target (new sampling rate = original rate / q_target)
        # For q_target = 1.5, the effective resampling factor is 1/1.5 ≈ 0.667.
        # We can represent 0.667 as 100/150.
        q_target = 1.5  # desired decimation factor
        up = 100      # Upsample factor
        down = int(100 * q_target)  # Downsample factor = 150 for q_target=1.5
        
        # Remove the mean and store it to add back later
        mean_original = np.mean(data)
        data_demeaned = data - mean_original
        
        # Upsample: Create an intermediate array by inserting zeros between original samples
        data_upsampled = np.zeros(len(data_demeaned) * up)
        data_upsampled[::up] = data_demeaned
        
        # Design an anti-aliasing FIR filter.
        # The normalized cutoff is set to 0.8 * min(1/up, 1/down)
        cutoff = 0.8 * min(1/up, 1/down)
        b = firwin(64, cutoff)
        # Filter the upsampled signal to remove spurious frequency components
        data_filtered = lfilter(b, 1, data_upsampled)
        
        # Downsample: Take every "down"-th sample from the filtered signal.
        data_resampled = data_filtered[::down] + mean_original
        
        # Create a new time axis for the resampled (decimated) data.
        dt = np.mean(np.diff(channel_time))
        # Since the new sampling rate is reduced by q_target, the new dt is:
        dt_new = dt * q_target
        channel_time_decimated = channel_time[0] + np.arange(len(data_resampled)) * dt_new
        
        # -------------------------------------------
        # FREQUENCY DOMAIN (Original and Resampled FFT)
        # -------------------------------------------
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - mean_original)
        magnitude = np.abs(fft_vals)

        f_axis_decimated = np.fft.rfftfreq(len(data_resampled), d=dt_new)
        fft_vals_decimated = np.fft.rfft(data_resampled - mean_original)
        magnitude_decimated = np.abs(fft_vals_decimated)

        ax_freq.plot(f_axis, magnitude,
                     label="FFT Mag - ORG",
                     color='blue',
                     zorder=10)
        ax_freq.plot(f_axis_decimated, magnitude_decimated,
                     label="FFT Mag - RESAMPLED",
                     color='red',
                     zorder=10)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
        ax_freq.grid(True)
        # ax_freq.set_xscale("log")
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
        
        # ------------------------------------------------------------
        # RECONSTRUCT RESAMPLED DATA VIA IFFT (Time Domain, Zero-Padding)
        # ------------------------------------------------------------
        data_reconstructed = np.fft.irfft(fft_vals_decimated, n=len(data)) + mean_original
        
        ax_recon.plot(channel_time, data_reconstructed,
                      linewidth=0.75,
                      label="Reconstructed Data",
                      rasterized=True,
                      zorder=10)
        ax_recon.set_xlabel("Time")
        ax_recon.set_ylabel(y_label)
        ax_recon.set_title(f"Channel: {channel.name} (Reconstructed TD)")
        ax_recon.grid(True)
        leg_recon = ax_recon.legend(loc="lower right")
        leg_recon.set_zorder(40)
        
        # -------------------------------------------------------------
        # DIFFERENCE SIGNAL (Time Domain: Original - Reconstructed)
        # -------------------------------------------------------------
        diff_signal = data - data_reconstructed
        
        ax_diff.plot(channel_time, diff_signal,
                     linewidth=0.75,
                     label="Diff (ORG - RECON)",
                     rasterized=True,
                     zorder=10)
        ax_diff.set_xlabel("Time")
        ax_diff.set_ylabel("Difference")
        ax_diff.set_title(f"Channel: {channel.name} (Difference TD)")
        ax_diff.grid(True)
        leg_diff = ax_diff.legend(loc="upper right")
        leg_diff.set_zorder(40)
        
        # --------------------------------------------------
        # FFT of the Difference Signal
        # --------------------------------------------------
        f_axis_diff = np.fft.rfftfreq(len(diff_signal), d=dt)
        fft_diff = np.fft.rfft(diff_signal)
        mag_diff = np.abs(fft_diff)
        ax_fft_diff.plot(f_axis_diff, mag_diff,
                         linewidth=0.75,
                         label="FFT Diff",
                         color='green',
                         rasterized=True,
                         zorder=10)
        ax_fft_diff.set_xlabel("Frequency (Hz)")
        ax_fft_diff.set_ylabel("Magnitude")
        ax_fft_diff.set_title(f"Channel: {channel.name} (FFT of Diff)")
        ax_fft_diff.grid(True)
        # ax_fft_diff.set_yscale("log")
        leg_fft_diff = ax_fft_diff.legend(loc="upper right")
        leg_fft_diff.set_zorder(40)
        
        # --------------------------------------------------
        # SFM Calculation for the Difference Signal (Value Only)
        # --------------------------------------------------
        eps = 1e-12
        power_spectrum = mag_diff**2
        geom_mean = np.exp(np.mean(np.log(power_spectrum + eps)))
        arith_mean = np.mean(power_spectrum)
        sfm = geom_mean / arith_mean
        
        # Display the SFM value as centered text (without a plot)
        ax_sfm.text(0.5, 0.5, f"SFM: {sfm:.3f}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax_sfm.transAxes,
                    fontsize=10)
        ax_sfm.axis("off")
    
    # Turn off unused axes for any rows that do not contain channel data.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")