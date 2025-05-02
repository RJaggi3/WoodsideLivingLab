import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
import pywt  # For wavelet transforms
import statistics

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name
file_name = "202503281215_SHM-2.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_wavelet.pdf")

# Read the TDMS file and collect all channels
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object so that each figure becomes one PDF page.
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

# Update grid: 6 rows x 3 columns (Time Domain, Frequency Domain, Histogram)
n_rows_fixed = 6
n_cols = 3

# Process channels in groups, up to 6 channels per page.
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    # Create a new figure with fixed grid, using A3 landscape size.
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Set the three axes for each row:
        # Column 0: Time Domain, Column 1: Frequency Domain, Column 2: Histogram.
        ax_time = axs[i, 0]
        ax_freq = axs[i, 1]
        ax_hist = axs[i, 2]
        
        # --- TIME DOMAIN PLOT ---
        data = channel[:]                   # Original raw TDMS data
        channel_time = channel.time_track()   # Corresponding time track
        x = np.array(channel_time)
        y = np.array(data)
        
        # Plot the original data in the time domain.
        ax_time.plot(channel_time, data,
                     label="Original Data",
                     linewidth=0.75,
                     rasterized=True,
                     zorder=10)
        
        # Compute and plot a trendline for the original data.
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            trend = p(x)
            ax_time.plot(x, trend,
                         'y-',
                         linewidth=2.0,
                         label="Trendline - Original",
                         zorder=30)
        
        # --- WAVELET TRANSFORM ANALYSIS ---
        # Use a 5-level discrete wavelet decomposition with the Daubechies 4 wavelet.
        coeffs = pywt.wavedec(data, 'db4', level=5)
        thresholded_coeffs = []
        # Count how many detail coefficients are thresholded (effectively eliminated).
        num_coeff_eliminated = 0
        total_detail_coeffs = 0
        for j, coeff in enumerate(coeffs):
            if j == 0:
                # Approximation coefficients are not thresholded.
                thresholded_coeffs.append(coeff)
            else:
                thr = np.std(coeff)
                thresh_coeff = pywt.threshold(coeff, thr, mode='soft')
                num_zeros = np.sum(np.abs(thresh_coeff) < 1e-12)  # Count nearly zero values
                num_coeff_eliminated += num_zeros
                total_detail_coeffs += len(coeff)
                thresholded_coeffs.append(thresh_coeff)
        # Reconstruct the signal using the thresholded coefficients.
        data_wavelet = pywt.waverec(thresholded_coeffs, 'db4')
        # Truncate if the reconstructed signal is longer than the original.
        data_wavelet = data_wavelet[:len(data)]
        
        # Calculate reduced coefficient count and percentage reduction.
        reduced_coeffs = total_detail_coeffs - num_coeff_eliminated
        if total_detail_coeffs > 0:
            eliminated_percentage = 100 * num_coeff_eliminated / total_detail_coeffs
        else:
            eliminated_percentage = 0
        print(f"Channel {channel.name}: Original data: {total_detail_coeffs}, Reduced data: {reduced_coeffs}, % reduction: {eliminated_percentage:.2f}%")
        
        # Plot the wavelet-reconstructed signal on the time domain plot (in red).
        ax_time.plot(channel_time, data_wavelet,
                     label="Wavelet Reconstruction",
                     linestyle='-.',
                     color='red',
                     zorder=25)
        
        # Statistics table comparing Original vs. Wavelet-reconstructed data.
        bound_lower      = np.min(data)
        bound_middle     = np.median(data)
        bound_upper      = np.max(data)
        bound_lower_wave = np.min(data_wavelet)
        bound_middle_wave = np.median(data_wavelet)
        bound_upper_wave = np.max(data_wavelet)
        table_data = [[f"{bound_lower:.6f}", f"{bound_middle:.6f}", f"{bound_upper:.6f}",
                       f"{bound_lower_wave:.6f}", f"{bound_middle_wave:.6f}", f"{bound_upper_wave:.6f}"]]
        col_labels = ["Lower\nORG", "Middle\nORG", "Upper\nORG",
                      "Lower\nWAVELET", "Middle\nWAVELET", "Upper\nWAVELET"]
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
        
        # Axes labels and title for the Time Domain plot.
        ax_time.set_xlabel("Time")
        y_label = channel.properties["unit_string"] if "unit_string" in channel.properties else "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right", fontsize=5)
        leg.set_zorder(40)
        
        # --- FREQUENCY DOMAIN PLOT ---
        dt = np.mean(np.diff(channel_time))  # Estimated sampling interval of original data.
        # FFT for original data.
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - np.mean(data))
        magnitude = np.abs(fft_vals)
        # FFT for wavelet-reconstructed signal.
        f_axis_wave = np.fft.rfftfreq(len(data_wavelet), d=dt)
        fft_vals_wave = np.fft.rfft(data_wavelet - np.mean(data_wavelet))
        magnitude_wave = np.abs(fft_vals_wave)
        
        ax_freq.plot(f_axis, magnitude,
                     label="FFT Magnitude - ORG",
                     color='blue',
                     zorder=10)
        ax_freq.plot(f_axis_wave, magnitude_wave,
                     label="FFT Magnitude - WAVELET",
                     color='red',  # Wavelet FFT plot now in red.
                     linestyle=':',
                     zorder=10)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
        ax_freq.grid(True)
        leg_freq = ax_freq.legend(loc="upper right", fontsize=5)
        leg_freq.set_zorder(40)
        
        # --- HISTOGRAM PLOT (Distribution) ---
        ax_hist.hist(data, bins=30, color='blue', alpha=0.50, label='Original')
        ax_hist.hist(data_wavelet, bins=30, color='red', alpha=0.50, label='Wavelet')
        # Compute means for both signals.
        mean_original_val = np.mean(data)
        mean_wavelet_val = np.mean(data_wavelet)
        threshold_val = 1e-3
        if abs(mean_original_val) < threshold_val:
            mean_original_str = f"{mean_original_val:.2e}"
        else:
            mean_original_str = f"{mean_original_val:.2f}"
        if abs(mean_wavelet_val) < threshold_val:
            mean_wavelet_str = f"{mean_wavelet_val:.2e}"
        else:
            mean_wavelet_str = f"{mean_wavelet_val:.2f}"
        # Plot vertical lines at the mean values.
        ax_hist.axvline(mean_original_val, color='blue', linestyle='dashed', linewidth=1.5,
                        label=f'Mean (Original)= {mean_original_str}')
        ax_hist.axvline(mean_wavelet_val, color='red', linestyle='dashed', linewidth=0.75,
                        label=f'Mean (Wavelet)= {mean_wavelet_str}')
        ax_hist.set_title(f"Channel: {channel.name} (Histogram)")
        ax_hist.set_xlabel("Measurement Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=5)
        ax_hist.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
    
    # Turn off unused axes for any rows that do not have a channel.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")