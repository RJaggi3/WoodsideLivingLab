import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import resample_poly  # Using polyphase filtering

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name.
file_name = "202109220920_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_POLYPHASE_trim.pdf")

# Read the TDMS file and collect all channels.
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object so each figure becomes one PDF page.
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

# Define grid: 6 rows x 6 columns
# Col 0: Time Domain (Original Data)
# Col 1: Frequency Domain (FFT of Original and Decimated)
# Col 2: Reconstructed Decimated Data (via polyphase upsampling)
# Col 3: Difference Signal (Original - Reconstructed) in Time Domain (trimmed)
# Col 4: FFT of the Trimmed Difference Signal (log-scale y-axis)
# Col 5: SFM for the Difference Signal (value only)
n_rows_fixed = 6
n_cols = 6

# Trim configuration (e.g. trim 5% from each edge)
trim_fraction = 0.05

for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(25, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Assign each subplot axis.
        ax_time     = axs[i, 0]
        ax_freq     = axs[i, 1]
        ax_recon    = axs[i, 2]
        ax_diff     = axs[i, 3]
        ax_fft_diff = axs[i, 4]
        ax_sfm      = axs[i, 5]
        
        # -----------------------------
        # TIME DOMAIN (Original Data)
        # -----------------------------
        data = channel[:]  # Raw TDMS data.
        channel_time = channel.time_track()  # Corresponding time track.
        
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
        # DECIMATION using Polyphase Filtering (resample_poly)
        # ---------------------------------------------------
        q = 2  # Decimation factor.
        mean_original = np.mean(data)
        # Subtract the mean, decimate, then add the mean back.
        data_decimated = resample_poly(data - mean_original, up=1, down=q) + mean_original
        channel_time_decimated = channel_time[::q]
        
        # -------------------------------------------
        # FREQUENCY DOMAIN (Original and Decimated FFT)
        # -------------------------------------------
        dt = np.mean(np.diff(channel_time))
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - mean_original)
        magnitude = np.abs(fft_vals)
        
        dt_decimated = np.mean(np.diff(channel_time_decimated))
        f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
        fft_vals_decimated = np.fft.rfft(data_decimated - mean_original)
        magnitude_decimated = np.abs(fft_vals_decimated)
        
        ax_freq.plot(f_axis, magnitude,
                     label="FFT Mag - ORG",
                     color='blue',
                     zorder=10)
        ax_freq.plot(f_axis_decimated, magnitude_decimated,
                     label="FFT Mag - DEC",
                     color='red',
                     zorder=10)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
        ax_freq.grid(True)
        ax_freq.set_yscale("log")
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
        
        # ------------------------------------------------------------
        # RECONSTRUCTION using Polyphase Filtering
        # ------------------------------------------------------------
        # Upsample the decimated signal to reconstruct it.
        data_reconstructed = resample_poly(data_decimated, up=q, down=1)
        # Ensure the reconstructed signal is the same length as the original.
        if len(data_reconstructed) > len(data):
            data_reconstructed = data_reconstructed[:len(data)]
        elif len(data_reconstructed) < len(data):
            data_reconstructed = np.pad(data_reconstructed, (0, len(data) - len(data_reconstructed)), mode='edge')
        
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
        # DIFFERENCE SIGNAL (Original - Reconstructed) (Time Domain)
        # -------------------------------------------------------------
        diff_signal = data - data_reconstructed
        
        # Trim edges of the difference signal.
        n_trim = int(trim_fraction * len(diff_signal))
        if len(diff_signal) > 2 * n_trim:
            diff_signal_trimmed = diff_signal[n_trim:-n_trim]
            time_trimmed = channel_time[n_trim:-n_trim]
        else:
            diff_signal_trimmed = diff_signal
            time_trimmed = channel_time
        
        ax_diff.plot(time_trimmed, diff_signal_trimmed,
                     linewidth=0.75,
                     label="Trimmed Diff (ORG - RECON)",
                     color="orange",
                     rasterized=True,
                     zorder=10)
        ax_diff.set_xlabel("Time")
        ax_diff.set_ylabel("Difference")
        ax_diff.set_title(f"Channel: {channel.name} (Diff TD; Trimmed)")
        ax_diff.grid(True)
        leg_diff = ax_diff.legend(loc="upper right")
        leg_diff.set_zorder(40)
        
        # --------------------------------------------------
        # FFT of the Trimmed Difference Signal
        # --------------------------------------------------
        # Use the trimmed difference signal for the FFT.
        f_axis_diff = np.fft.rfftfreq(len(diff_signal_trimmed), d=dt)
        fft_diff = np.fft.rfft(diff_signal_trimmed)
        mag_diff = np.abs(fft_diff)
        ax_fft_diff.plot(f_axis_diff, mag_diff,
                         linewidth=0.75,
                         label="FFT Diff (Trimmed)",
                         color='green',
                         rasterized=True,
                         zorder=10)
        ax_fft_diff.set_xlabel("Frequency (Hz)")
        ax_fft_diff.set_ylabel("Magnitude")
        ax_fft_diff.set_title(f"Channel: {channel.name} (FFT of Diff; Trimmed)")
        ax_fft_diff.grid(True)
        ax_fft_diff.set_yscale("log")
        leg_fft_diff = ax_fft_diff.legend(loc="upper right")
        leg_fft_diff.set_zorder(40)
        
        # --------------------------------------------------
        # SFM Calculation for the Trimmed Difference Signal (Value Only)
        # --------------------------------------------------
        eps = 1e-12
        power_spectrum = mag_diff**2
        geom_mean = np.exp(np.mean(np.log(power_spectrum + eps)))
        arith_mean = np.mean(power_spectrum)
        sfm = geom_mean / arith_mean
        
        # Display the SFM value as centered text.
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