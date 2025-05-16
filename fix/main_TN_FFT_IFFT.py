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

# Define your TDMS file and output PDF name.
file_name = "202305282340_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_TD_diff_IFFT.pdf")

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

# Define grid dimensions: 6 rows x 4 columns.
n_rows_fixed = 6
n_cols = 4

# Process channels in groups, up to 6 channels per page.
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    # Create a new figure with fixed grid, using A3 landscape size.
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Column 0: Raw Time Domain (with trendline & stats).
        ax_time = axs[i, 0]
        # Column 1: FFT Overlay for Raw and Decimated Data.
        ax_freq = axs[i, 1]
        # Column 2: Time Domain for Difference Signal (using FFT-based reconstruction).
        ax_diff_time = axs[i, 2]
        # Column 3: FFT for the Difference Signal.
        ax_diff_freq = axs[i, 3]
        
        # ----- TIME DOMAIN PLOT FOR RAW DATA -----
        data = channel[:]                         # raw TDMS data
        channel_time = channel.time_track()         # corresponding time track
        x = np.array(channel_time)
        y = np.array(data)
        
        # Plot the raw data.
        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Original Data",
                     rasterized=True,
                     zorder=10)
        
        # Compute and plot a linear trendline.
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trend = np.poly1d(coeffs)(x)
            ax_time.plot(x, trend,
                         'y-',
                         linewidth=2.0,
                         label="Trendline",
                         zorder=30)
        
        # Insert a statistics table for raw data (min, median, max).
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
        
        # ----- FFT OVERLAY PLOT FOR RAW & DECIMATED DATA (Column 1) -----
        # Compute raw data FFT.
        dt = np.mean(np.diff(channel_time))  # estimated sampling interval for raw data
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - np.mean(data))
        magnitude = np.abs(fft_vals)
        
        # Compute decimated data and its FFT.
        q = 2  # Decimation factor.
        mean_original = np.mean(data)
        data_decimated = decimate(data - mean_original, q, ftype='iir', zero_phase=True) + mean_original
        dt_decimated = q * dt  # effective sampling interval of decimated signal
        f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
        fft_decimated = np.fft.rfft(data_decimated - np.mean(data_decimated))
        magnitude_decimated = np.abs(fft_decimated)
        
        # Plot overlay: FFT of raw data and decimated data.
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
        
        # ----- FFT-BASED RECONSTRUCTION AND DIFFERENCE SIGNAL (Time Domain) -----
        # Reconstruct the decimated signal to full length using FFT-based zero-padding.
        data_reconstructed = fft_reconstruct(data_decimated, len(data))
        # Compute the difference between the original data and the reconstructed decimated signal.
        diff_signal = data - data_reconstructed
        
        ax_diff_time.plot(channel_time, diff_signal,
                          linewidth=0.75,
                          label="Difference (Original - Reconstructed)",
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
        # Compute the FFT of the full-resolution difference signal.
        dt_diff = dt  # using the original sampling interval
        f_diff = np.fft.rfftfreq(len(diff_signal), d=dt_diff)
        fft_diff = np.fft.rfft(diff_signal - np.mean(diff_signal))
        magnitude_diff = np.abs(fft_diff)
        
        ax_diff_freq.plot(f_diff, magnitude_diff,
                          label="FFT Diff",
                          color='red',
                          zorder=10)
        ax_diff_freq.set_xlabel("Frequency (Hz)")
        ax_diff_freq.set_ylabel("Magnitude")
        ax_diff_freq.set_title(f"Channel: {channel.name} (FFT of Diff)")
        ax_diff_freq.grid(True)
        leg_diff_freq = ax_diff_freq.legend(loc="upper right")
        leg_diff_freq.set_zorder(40)
        
    # Turn off unused axes for rows that have no channel data.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")