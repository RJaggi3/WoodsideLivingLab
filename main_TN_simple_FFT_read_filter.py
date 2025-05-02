import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import find_peaks

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name
file_name = "202305282340_SHM-6.tdms"
pdf_name = file_name.replace(".tdms", "_simple_FFT_denoised_table.pdf")

# Read the TDMS file and collect all channels
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object to compile multiple pages into one PDF.
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

# Define a fixed grid with 6 rows x 4 columns:
# Column 0: Raw Time Domain, Column 1: Frequency Domain,
# Column 2: Denoised Time Domain (with reduction % in title),
# Column 3: Table displaying metrics.
n_rows_fixed = 6
n_cols = 4

# Adjusted figure size for 4 columns (width scaled accordingly)
figsize = (33, 11.69)  # roughly 8.25 inches per column

for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=figsize)
    
    # In case of a single row, wrap axs in a numpy.array for consistency.
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    for i, channel in enumerate(channel_group):
        # Define subplot axes for each section.
        ax_time     = axs[i, 0]  # Raw time-domain plot with trendline.
        ax_freq     = axs[i, 1]  # Frequency domain (FFT) plot with detected peaks.
        ax_denoised = axs[i, 2]  # Denoised time-domain plot (after inverse FFT).
        ax_table    = axs[i, 3]  # Table for metrics.

        # --- RAW TIME DOMAIN PLOT ---
        data = np.array(channel[:])
        time_track = np.array(channel.time_track())
        ax_time.plot(time_track, data, linewidth=0.75,
                     label="Raw Data", rasterized=True, zorder=10)
        if len(time_track) > 1:
            coeffs = np.polyfit(time_track, data, 1)
            p = np.poly1d(coeffs)
            trend = p(time_track)
            ax_time.plot(time_track, trend, 'y-', linewidth=2.0,
                         label="Trendline", zorder=30)
        ax_time.set_xlabel("Time")
        y_label = channel.properties.get("unit_string", "Measurement Value")
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name}\n(Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # --- FREQUENCY DOMAIN CALCULATIONS ---
        dt = np.mean(np.diff(time_track))
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        mean_val = np.mean(data)
        fft_vals = np.fft.rfft(data - mean_val)
        magnitude = np.abs(fft_vals)
        
        # Detect peaks using a threshold (10% of max magnitude).
        threshold = np.max(magnitude) * 0.1  
        peaks, _ = find_peaks(magnitude, height=threshold)
        
        ax_freq.plot(f_axis, magnitude, label="FFT Magnitude",
                     color='blue', zorder=10)
        ax_freq.plot(f_axis[peaks], magnitude[peaks], 'ro', markersize=3,
                     label="Peaks", zorder=20)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name}\n(Frequency Domain)")
        ax_freq.grid(True)
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
        
        # --- NOISE DECIMATION & INVERSE FFT ---
        fft_decimated = np.zeros_like(fft_vals)
        window_size = 2  # keeps a small window around each detected peak
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(fft_vals), peak + window_size + 1)
            fft_decimated[start:end] = fft_vals[start:end]
        
        # Reconstruct denoised signal via inverse FFT and add back the mean.
        data_denoised = np.fft.irfft(fft_decimated, n=len(data)) + mean_val
        
        # --- METRIC CALCULATIONS ---
        # Calculate the data reduction in the frequency domain.
        num_nonzero = np.count_nonzero(fft_decimated)
        total_bins = len(fft_vals)
        retained_percentage = (num_nonzero / total_bins) * 100
        reduction_percentage = 100 - retained_percentage  # percentage of bins zeroed out
        
        # Calculate absolute differences in the high and low bounds.
        orig_max = np.max(data)
        orig_min = np.min(data)
        denoised_max = np.max(data_denoised)
        denoised_min = np.min(data_denoised)
        high_bound_diff = orig_max - denoised_max
        low_bound_diff  = orig_min - denoised_min
        
        # Calculate percentage differences relative to the original high/low.
        # (If the original bound is zero, default the percentage difference to 0.)
        high_diff_percentage = (abs(high_bound_diff) / abs(orig_max) * 100) if orig_max != 0 else 0
        low_diff_percentage  = (abs(low_bound_diff) / abs(orig_min) * 100) if orig_min != 0 else 0
        
        # --- PLOT DENOISED TIME DOMAIN ---
        ax_denoised.plot(time_track, data_denoised, linewidth=0.75,
                         color='green', label="Inverse FFT Signal", rasterized=True, zorder=10)
        ax_denoised.set_xlabel("Time")
        ax_denoised.set_ylabel(y_label)
        ax_denoised.set_title(f"Channel: {channel.name}\n(Reconstructed Time Domain)")
        ax_denoised.grid(True)
        leg_denoised = ax_denoised.legend(loc="lower right")
        leg_denoised.set_zorder(40)
        
        # --- OUTPUT TABLE OF METRICS IN COLUMN 4 ---
        ax_table.axis('tight')
        ax_table.axis('off')
        # Prepare table data with two rows:
        # Row labels: "Absolute" and "Percentage" to indicate value type.
        # For the "Data Reduction" column, the value is only applicable in the first row.
        table_data = [
            [f"{reduction_percentage:.2f}%", f"{high_bound_diff:.4g}", f"{low_bound_diff:.4g}"],
            ["-", f"{high_diff_percentage:.2f}%", f"{low_diff_percentage:.2f}%"]
        ]
        row_labels = ["Absolute", "Percentage"]
        col_labels = ["Data Reduction", "High Bound Diff", "Low Bound Diff"]
        
        table = ax_table.table(cellText=table_data,
                               colLabels=col_labels,
                               rowLabels=row_labels,
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax_table.set_title("Signal Differences", fontsize=8)
    
    # Turn off unused subplots for any remaining rows.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots with noise decimation and metric tables have been compiled into the file: {pdf_name}")