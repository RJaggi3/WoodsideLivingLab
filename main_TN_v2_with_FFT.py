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
pdf_name = file_name.replace(".tdms", "_FREQ_v2.pdf")

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

# We'll always create a fixed grid: 6 rows x 2 columns (even if some rows will be blank)
n_rows_fixed = 6
n_cols = 2

# Process channels in groups, up to 6 channels per page
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    # Create a new figure with fixed grid, using A3 landscape size (16.54" x 11.69")
    fig, axs = plt.subplots(nrows=n_rows_fixed,
                            ncols=n_cols,
                            figsize=(16.54, 11.69))
    # Ensure axs is a 2D array
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    # Loop over each available channel and plot:
    for i, channel in enumerate(channel_group):
        # Prepare the two axes in row i: left for time domain, right for frequency domain.
        ax_time = axs[i, 0]
        ax_freq = axs[i, 1]
        
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
        
        # Decimation
        q = 2  # Decimation factor
        mean_original = np.mean(data)
        data_demeaned = data - mean_original
        data_decimated = decimate(data_demeaned, q, ftype='iir', zero_phase=True)
        data_decimated += mean_original
        channel_time_decimated = channel_time[::q]
        
        # Calculate and print how many samples were deleted after decimation
        original_length = len(data)
        decimated_length = len(data_decimated)
        deleted_samples = original_length - decimated_length
        percent_deleted = 100 * deleted_samples / original_length
        print(f"Channel {channel.name}: Original samples = {original_length}, "
              f"Decimated samples = {decimated_length}, "
              f"Deleted samples = {deleted_samples} ({percent_deleted:.2f}% reduction)")
        
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
        
        # Statistics table (for both original and decimated data)
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
        if "unit_string" in channel.properties:
            y_label = channel.properties["unit_string"]
        else:
            y_label = "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # --- FREQUENCY DOMAIN PLOT (Dynamic Analysis) ---
        # For frequency analysis, we compute the FFT on the demeaned signal.
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
        leg_freq = ax_freq.legend(loc="lower right")
        leg_freq.set_zorder(40)
    
    # Turn off unused axes for rows that have no channel data.
    for j in range(n_channels, n_rows_fixed):
        axs[j, 0].axis("off")
        axs[j, 1].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")