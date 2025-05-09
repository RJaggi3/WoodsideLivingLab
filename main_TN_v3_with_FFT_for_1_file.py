import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate
import statistics

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name
file_name = "202503281215_SHM-2.tdms"
pdf_name = file_name.replace(".tdms", "_FREQ_v3.pdf")

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

# Update grid: 6 rows x 4 columns to accommodate the additional histogram column.
n_rows_fixed = 6
n_cols = 4

# Process channels in groups, up to 6 channels per page
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    # Create a new figure with fixed grid, using A3 landscape size (16.54" x 11.69")
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    # Ensure axs is a 2D array even if n_rows_fixed == 1
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    # Loop over each available channel and plot:
    for i, channel in enumerate(channel_group):
        # Prepare the four axes in row i:
        # Column 0: Time Domain, Column 1: Frequency Domain,
        # Column 2: Box Plot, Column 3: Histogram Plot.
        ax_time = axs[i, 0]
        ax_freq = axs[i, 1]
        ax_box  = axs[i, 2]
        ax_hist = axs[i, 3]
        
        # --- TIME DOMAIN PLOT (Static Analysis) ---
        data = channel[:]                        # raw TDMS data
        channel_time = channel.time_track()        # corresponding time track
        x = np.array(channel_time)
        y = np.array(data)
        
        # Plot original data
        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Original Data",
                     rasterized=True,
                     zorder=10)
        
        # Compute and plot trendline for original data
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            trend = p(x)
            ax_time.plot(x, trend,
                         'y-',
                         linewidth=2.0,
                         label="Trendline - Original",
                         zorder=30)
        
        # --- DECIMATION ---
        q = 2  # Decimation factor
        mean_original = np.mean(data)
        data_demeaned = data - mean_original
        data_decimated = decimate(data_demeaned, q, ftype='iir', zero_phase=True)
        data_decimated += mean_original
        channel_time_decimated = channel_time[::q]
        
        # Calculate decimation statistics
        original_length = len(data)
        decimated_length = len(data_decimated)
        deleted_samples = original_length - decimated_length
        percent_deleted = 100 * deleted_samples / original_length
        print(f"Channel {channel.name}: Original samples = {original_length}, "
              f"Decimated samples = {decimated_length}, "
              f"Deleted samples = {deleted_samples} ({percent_deleted:.2f}% reduction)")
        
        # Plot the decimated signal on the time domain plot
        ax_time.plot(channel_time_decimated, data_decimated,
                     label="Decimated Signal",
                     color='red',
                     linestyle='--',
                     rasterized=True,
                     zorder=20)
        
        # Trendline for decimated data
        x_decimate = np.array(channel_time_decimated)
        y_decimate = np.array(data_decimated)
        if len(x_decimate) > 1:
            coeffs_deci = np.polyfit(x_decimate, y_decimate, 1)
            p_deci = np.poly1d(coeffs_deci)
            trend_decimate = p_deci(x_decimate)
            ax_time.plot(x_decimate, trend_decimate,
                         'b--',
                         linewidth=0.7,
                         label="Trendline - Decimate",
                         zorder=30)
        
        # Statistics table for Time Domain plot (both original and decimated)
        bound_lower      = np.min(data)
        bound_middle     = np.median(data)
        bound_upper      = np.max(data)
        bound_lower_deci = np.min(data_decimated)
        bound_middle_deci= np.median(data_decimated)
        bound_upper_deci = np.max(data_decimated)
        table_data = [[f"{bound_lower:.6f}", f"{bound_middle:.6f}", f"{bound_upper:.6f}",
                       f"{bound_lower_deci:.6f}", f"{bound_middle_deci:.6f}", f"{bound_upper_deci:.6f}"]]
        col_labels = ["Lower\nORG", "Middle\nORG", "Upper\nORG",
                      "Lower\nDEC", "Middle\nDEC", "Upper\nDEC"]
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
        
        # Axes labels and title for Time Domain plot
        ax_time.set_xlabel("Time")
        y_label = channel.properties["unit_string"] if "unit_string" in channel.properties else "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # --- FREQUENCY DOMAIN PLOT (Dynamic Analysis) ---
        dt = np.mean(np.diff(channel_time))  # estimated sampling interval
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        fft_vals = np.fft.rfft(data - mean_original)
        magnitude = np.abs(fft_vals)
        
        dt_decimated = np.mean(np.diff(channel_time_decimated))
        f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
        fft_vals_decimated = np.fft.rfft(data_decimated - mean_original)
        magnitude_decimated = np.abs(fft_vals_decimated)
        
        ax_freq.plot(f_axis, magnitude,
                     label="FFT Magnitude - ORG",
                     color='blue',
                     zorder=10)
        ax_freq.plot(f_axis_decimated, magnitude_decimated,
                     label="FFT Magnitude - DEC",
                     color='red',
                     zorder=10)
        
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
        ax_freq.grid(True)
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
        
        # --- BOX PLOT (Comparing Original vs. Decimated) ---
        ax_box.boxplot([data, data_decimated], tick_labels=["Original", "Decimated"])
        ax_box.set_title(f"Channel: {channel.name} (Box Plot)")
        ax_box.set_ylabel("Measurement Value")
        ax_box.grid(True)
        
        # --- HISTOGRAM PLOT (Distribution with Mean) ---
        # Overlay histograms for original and decimated data using a semi-transparent style.
        ax_hist.hist(data, bins=30, color='blue', alpha=0.5, label='Original')
        ax_hist.hist(data_decimated, bins=30, color='red', alpha=0.5, label='Decimated')
        # Compute the mean on the histogram
        mean_original_val = np.mean(data)
        mean_decimated_val = np.mean(data_decimated)
        
        # Set threshold for considering a value "really small"
        threshold = 1e-3
        if abs(mean_original_val) < threshold:
            mean_original_str = f"{mean_original_val:.2e}"
        else:
            mean_original_str = f"{mean_original_val:.2f}"
        if abs(mean_decimated_val) < threshold:
            mean_decimated_str = f"{mean_decimated_val:.2e}"
        else:
            mean_decimated_str = f"{mean_decimated_val:.2f}"
        
        # Plot vertical lines for the means with appropriate formatting in the legend.
        ax_hist.axvline(mean_original_val, color='blue', linestyle='dashed', linewidth=1.5,
                        label=f'Mean (Original)= {mean_original_str}')
        ax_hist.axvline(mean_decimated_val, color='red', linestyle='dashed', linewidth=0.75,
                        label=f'Mean (Decimated)= {mean_decimated_str}')
        ax_hist.set_title(f"Channel: {channel.name} (Histogram)")
        ax_hist.set_xlabel("Measurement Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=5)
        # Adjust x-axis tick formatting: if the measurement values are very small, scientific notation will be used.
        ax_hist.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
    
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