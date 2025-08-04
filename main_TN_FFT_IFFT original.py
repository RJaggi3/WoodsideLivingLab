import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate
import statistics

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fft_reconstruct(decimated_signal, target_length):
    """
    Reconstruct a full-length time domain signal from a decimated signal using FFT-based zero-padding.

    Parameters:
      decimated_signal: The decimated signal array (of length M).
      target_length: The desired length (N) of the reconstructed signal.
      
    Returns:
      A time-domain signal of length target_length, obtained by zero-padding the FFT of the decimated signal.
    """
    N_dec = len(decimated_signal)
    fft_decimated = np.fft.fft(decimated_signal)
    padded_fft = np.zeros(target_length, dtype=complex)
    
    # Handle even and odd lengths appropriately to preserve symmetry
    if N_dec % 2 == 0:
        half = N_dec // 2
        padded_fft[:half] = fft_decimated[:half]
        padded_fft[-half:] = fft_decimated[half:]
    else:
        half = N_dec // 2
        padded_fft[:half+1] = fft_decimated[:half+1]
        padded_fft[-half:] = fft_decimated[half+1:]
    
    reconstructed_signal = np.fft.ifft(padded_fft).real
    return reconstructed_signal


def fft_reconstruct_original(original_signal, target_length, decimation_factor):
    """
    Reconstruct a lower-resolution version of the original signal by mapping its FFT bins
    into a new FFT array of length target_length (which should equal the length of the decimated data).
    
    The new FFT array is built by picking every decimation_factor-th FFT coefficient from the original,
    placing that coefficient in the corresponding position and setting all other bins to zero. 
    The inverse FFT of this new array yields a time-domain signal on the decimated grid.
    
    Parameters:
      original_signal: Original time-domain signal (length N_orig)
      target_length: Desired output length (should equal len(decimated_data), i.e., N_dec)
      decimation_factor: The decimation factor (q), so ideally N_orig = q * target_length
      
    Returns:
      A time-domain signal (via IFFT) of length target_length reconstructed from the selected FFT bins.
    """
    N_orig = len(original_signal)
    fft_original = np.fft.fft(original_signal)  # FFT of the original signal
    N_target = target_length  # This should be equal to len(data_decimated)
    
    # Create an empty FFT array of length N_target.
    new_fft = np.zeros(N_target, dtype=complex)
    
    # For real signals, the number of nonredundant (positive) FFT bins in the target is:
    pos_bins = N_target // 2 + 1

    # Map the positive frequency bins:
    for i in range(pos_bins):
        idx = i * decimation_factor
        if idx < N_orig:
            new_fft[i] = fft_original[idx]
    
    # Map the negative frequency bins:
    for i in range(1, N_target - pos_bins + 1):
        new_fft[-i] = fft_original[-(i * decimation_factor)]
    
    # Compute the inverse FFT to get the time-domain signal on the decimated grid.
    reconstructed_signal = np.fft.ifft(new_fft).real
    return reconstructed_signal


# Define your TDMS file and output PDF name.
file_name = "202503281215_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_TD_diff_IFFT_ORG2.pdf")

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

# Define grid dimensions: 6 rows x 5 columns.
n_rows_fixed = 6
n_cols = 5  # Column 0: raw time, Column 1: FFT overlay, Column 2: diff signal, Column 3: FFT diff, Column 4: SFM

# Set decimation factor (q)
decimation_factor = 2  # Adjust as needed

# Process channels in groups (up to 6 channels per page).
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Set up subplots:
        # Column 0: Raw time-domain.
        ax_time = axs[i, 0]
        # Column 1: FFT overlay (raw and decimated).
        ax_freq = axs[i, 1]
        # Column 2: Difference signal on the decimated grid.
        ax_diff_time = axs[i, 2]
        # Column 3: FFT of the difference signal.
        ax_diff_fft = axs[i, 3]
        # Column 4: Spectral Flatness Measure display.
        ax_sf = axs[i, 4]
        
        # ----- TIME DOMAIN PLOT FOR RAW DATA -----
        data = channel[:]                         # Raw TDMS data.
        channel_time = channel.time_track()         # Original time track.
        x = np.array(channel_time)
        y = np.array(data)
        
        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Original Data",
                     rasterized=True,
                     zorder=10)
        
        # Compute and plot linear trendline.
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)(x)
            ax_time.plot(x, trend,
                         'y-',
                         linewidth=2.0,
                         label="Trendline",
                         zorder=30)
        
        # Insert statistics table (min, median, max).
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
        
        # ----- COMPUTE DECIMATED DATA (CALCULATED ONCE) -----
        dt = np.mean(np.diff(channel_time))  # Sampling interval for raw data.
        data_mean = np.mean(data)
        centered_data = data - data_mean
        data_decimated = decimate(centered_data, decimation_factor, ftype='iir', zero_phase=True) + data_mean
        dt_decimated = decimation_factor * dt  # New sampling interval.
        time_decimated = channel_time[::decimation_factor]  # Downsampled time vector.
        
        # Compute FFT parameters for decimated data.
        f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
        fft_decimated = np.fft.rfft(data_decimated - np.mean(data_decimated))
        magnitude_decimated = np.abs(fft_decimated)
        
        # ----- FFT OVERLAY PLOT (Column 1) -----
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
        # Downsample (reconstruct) the original signal using its FFT bins so that its length equals that of the decimated data.
        original_downsampled = fft_reconstruct_original(data, len(data_decimated), decimation_factor)
        # Compute the difference signal on the decimated grid.
        diff_signal = original_downsampled - data
        
        ax_diff_time.plot(time_decimated, diff_signal,
                          linewidth=0.75,
                          label="Diff (Orig_downsampled - Decimated)",
                          color='red',
                          rasterized=True,
                          zorder=10)
        ax_diff_time.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax_diff_time.set_xlabel("Time (Decimated)")
        ax_diff_time.set_ylabel("Difference")
        ax_diff_time.set_title(f"Channel: {channel.name} (Diff Signal)")
        ax_diff_time.grid(True)
        leg_diff = ax_diff_time.legend(loc="lower right")
        leg_diff.set_zorder(40)
        
        # ----- FFT OF THE DIFFERENCE SIGNAL (Column 3) -----
        # Now that the diff signal is on the decimated grid, use dt_decimated.
        f_diff = np.fft.rfftfreq(len(diff_signal), d=dt_decimated)
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
        # Compute the power spectrum from the difference signal FFT.
        power_spec = magnitude_diff**2
        eps = 1e-12  # small constant to prevent log(0)
        sfm = np.exp(np.mean(np.log(power_spec + eps))) / np.mean(power_spec + eps)
        # Provide a simple interpretation.
        interpretation = "Flat (White Noise)" if sfm > 0.8 else "Tonal"
        
        ax_sf.set_title(f"Channel: {channel.name}\nSpectral Flatness")
        ax_sf.text(0.5, 0.5, f"SFM = {sfm:.3f}\n{interpretation}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax_sf.transAxes,
                   fontsize=8)
        ax_sf.set_xticks([])
        ax_sf.set_yticks([])
        ax_sf.set_frame_on(False)
        
    # Turn off any unused axes if fewer than n_rows_fixed channels were processed.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")