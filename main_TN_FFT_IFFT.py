import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate
import statistics

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fft_reconstruct(decimated_signal, target_length, decimation_factor):
    """
    Reconstruct a full-length time domain signal from a decimated signal by
    mapping its FFT bins into an FFT of length `target_length` and zeroing 
    any missing bins.
    
    Parameters:
      decimated_signal: The decimated signal array (length M)
      target_length: The desired output length (should be equal to the original signal’s length)
      decimation_factor: The decimation factor (q)
      
    Returns:
      A time-domain signal of length target_length, reconstructed using the 
      FFT bins from the decimated signal.
    """
    M = len(decimated_signal)
    fft_decimated = np.fft.fft(decimated_signal)  # FFT of decimated signal (length M)
    N = target_length

    # Optional check: ideally, original length should equal decimation_factor * M.
    #if N != decimation_factor * M:
    #    print("Warning: target_length != decimation_factor * len(decimated_signal)")
    
    # Create an empty FFT array of the full (original) length.
    full_fft = np.zeros(N, dtype=complex)
    
    # Number of positive-frequency bins for a real signal is M//2 + 1.
    pos_bins = M // 2 + 1

    # Map the positive frequencies
    for i in range(pos_bins):
        idx = i * decimation_factor  # corresponding FFT bin index in the original grid
        if idx < N:
            full_fft[idx] = fft_decimated[i]
    
    # Map the negative frequencies.
    for i in range(1, M - pos_bins + 1):
        full_fft[-(i * decimation_factor)] = fft_decimated[M - i]
    
    reconstructed_signal = np.fft.ifft(full_fft).real
    return reconstructed_signal

# Define your TDMS file and output PDF name.
file_name = "202109220920_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_TD_diff_IFFT_3.pdf")

# Read the TDMS file and collect all channels.
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object so that each figure becomes one PDF page.
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

# Define grid dimensions: 6 rows x 5 columns.
n_rows_fixed = 6
n_cols = 5  # updated to include an extra column for SFM

# Set a decimation factor (q)
decimation_factor = 2  # or any integer decimation factor you're using

# Process channels in groups (up to 6 channels per page).
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Set up subplots:
        # Column 0: raw time-domain.
        ax_time = axs[i, 0]
        # Column 1: FFT overlay (raw and decimated).
        ax_freq = axs[i, 1]
        # Column 2: time-domain difference signal.
        ax_diff_time = axs[i, 2]
        # Column 3: FFT of the difference signal.
        ax_diff_fft = axs[i, 3]
        # Column 4: Spectral Flatness Measure display.
        ax_sf = axs[i, 4]
        
        # ----- TIME DOMAIN PLOT FOR RAW DATA -----
        data = channel[:]  # Raw TDMS data.
        channel_time = channel.time_track()  # Corresponding time track.
        x = np.array(channel_time)
        y = np.array(data)
        
        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Original Data",
                     rasterized=True,
                     zorder=10)
        
        # Plot a linear trendline.
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)(x)
            ax_time.plot(x, trend,
                         'y-',
                         linewidth=2.0,
                         label="Trendline",
                         zorder=30)
        
        # Insert a statistics table (min, median, max).
        bound_lower = np.min(data)
        bound_middle = np.median(data)
        bound_upper = np.max(data)
        table_data = [[f"{bound_lower:.6f}", f"{bound_middle:.6f}", f"{bound_upper:.6f}"]]
        col_labels = ["Lower", "Median", "Upper"]
        the_table = ax_time.table(cellText=table_data,
                                  colLabels=col_labels,
                                  loc='upper center',
                                  cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(4)
        the_table.scale(1, 0.8)
        for key, cell in the_table.get_celld().items():
            cell.set_facecolor('white')
            cell.set_alpha(0.8)
            cell.set_edgecolor('black')
            cell.get_text().set_color('black')
        the_table.set_zorder(40)
        
        ax_time.set_xlabel("Time")
        y_label = channel.properties["unit_string"] if "unit_string" in channel.properties else "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # ----- COMPUTE DECIMATED DATA (CALCULATED ONLY ONCE) -----
        dt = np.mean(np.diff(channel_time))  # Sampling interval for raw data.
        data_mean = np.mean(data)
        centered_data = data - data_mean
        data_decimated = decimate(centered_data, decimation_factor, ftype='iir', zero_phase=True) + data_mean
        dt_decimated = decimation_factor * dt
        
        # FFT parameters for decimated data.
        f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
        fft_decimated = np.fft.rfft(data_decimated - np.mean(data_decimated))
        magnitude_decimated = np.abs(fft_decimated)
        
        # ----- FFT OVERLAY PLOT (Column 1) -----
        # FFT of raw data.
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - np.mean(data))
        magnitude = np.abs(fft_vals)
        
        ax_freq.plot(f_axis, magnitude,
                     label="FFT Raw",
                     color='blue',
                     zorder=10)
        ax_freq.plot(f_axis_decimated, magnitude_decimated,
                     label="FFT Decimated",
                     color='red',
                     zorder=10)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (FFT Overlay)")
        ax_freq.grid(True)
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
        
        # ----- FFT-BASED RECONSTRUCTION & DIFFERENCE SIGNAL (Column 2) -----
        data_reconstructed = fft_reconstruct(data_decimated, len(data), decimation_factor)
        diff_signal = data - data_reconstructed
        
        ax_diff_time.plot(channel_time, diff_signal,
                          linewidth=0.75,
                          label="Diff (Orig - Recon)",
                          color='red',
                          rasterized=True,
                          zorder=10)
        ax_diff_time.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax_diff_time.set_xlabel("Time")
        ax_diff_time.set_ylabel("Difference")
        ax_diff_time.set_title(f"Channel: {channel.name} (Diff Signal)")
        ax_diff_time.grid(True)
        leg_diff = ax_diff_time.legend(loc="lower right")
        leg_diff.set_zorder(40)
        
        # ----- FFT OF THE DIFFERENCE SIGNAL (Column 3) -----
        dt_diff = dt  # Use the original sampling interval.
        f_diff = np.fft.rfftfreq(len(diff_signal), d=dt_diff)
        fft_diff = np.fft.rfft(diff_signal - np.mean(diff_signal))
        magnitude_diff = np.abs(fft_diff)
        
        ax_diff_fft.plot(f_diff, magnitude_diff,
                         label="FFT Diff",
                         color='red',
                         zorder=10)
        ax_diff_fft.set_xlabel("Frequency (Hz)")
        ax_diff_fft.set_ylabel("Magnitude")
        ax_diff_fft.set_title(f"Channel: {channel.name} (FFT of Diff)")
        ax_diff_fft.grid(True)
        leg_diff_fft = ax_diff_fft.legend(loc="upper right")
        leg_diff_fft.set_zorder(40)
        
        # ----- SPECTRAL FLATNESS MEASURE (Column 4) -----
        # The power spectrum calculated from the diff signal FFT:
        power_spec = magnitude_diff**2
        eps = 1e-12  # small constant to avoid log(0)
        sfm = np.exp(np.mean(np.log(power_spec + eps))) / np.mean(power_spec + eps)
        # Optional: Provide an interpretation.
        if sfm > 0.8:
            interpretation = "Flat (White Noise) >0.8 "
        else:
            interpretation = "Tonal <0.8"
        
        ax_sf.set_title(f"Channel: {channel.name}\nSpectral Flatness")
        # Display the SFM value and interpretation in the axis.
        ax_sf.text(0.5, 0.5, f"SFM = {sfm:.3f}\n{interpretation}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax_sf.transAxes,
                   fontsize=8)
        # Remove axis ticks.
        ax_sf.set_xticks([])
        ax_sf.set_yticks([])
        ax_sf.set_frame_on(False)
        
    # Turn off any unused axes.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")