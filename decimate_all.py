import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate
import statistics
import os
import glob

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Folder containing the TDMS files
tdms_folder = "tdms_files"

# Get all .tdms files in the folder
tdms_files = glob.glob(os.path.join(tdms_folder, "*.tdms"))

for file_path in tdms_files:
    file_name = os.path.basename(file_path)
    pdf_name = os.path.join(tdms_folder, file_name.replace(".tdms", "_FREQ_v3.pdf"))

    print(f"\nProcessing {file_name}...")

    tdms_file = TdmsFile.read(file_path)
    all_groups = tdms_file.groups()
    channels = []
    for group in all_groups:
        for channel in group.channels():
            channels.append(channel)

    plt.rcParams.update({'font.size': 7})
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    n_rows_fixed = 6
    n_cols = 4

    for channel_group in chunks(channels, 6):
        n_channels = len(channel_group)
        fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
        if n_rows_fixed == 1:
            axs = np.array([axs])
        
        for i, channel in enumerate(channel_group):
            ax_time = axs[i, 0]
            ax_freq = axs[i, 1]
            ax_box  = axs[i, 2]
            ax_hist = axs[i, 3]

            data = channel[:]
            channel_time = channel.time_track()
            x = np.array(channel_time)
            y = np.array(data)

            ax_time.plot(x, y, linewidth=0.75, label="Original Data", rasterized=True, zorder=10)
            if len(x) > 1:
                coeffs = np.polyfit(x, y, 1)
                ax_time.plot(x, np.poly1d(coeffs)(x), 'y-', linewidth=2.0, label="Trendline - Original", zorder=30)

            q = 2
            mean_original = np.mean(data)
            data_demeaned = data - mean_original
            data_decimated = decimate(data_demeaned, q, ftype='iir', zero_phase=True) + mean_original
            channel_time_decimated = channel_time[::q]

            x_decimate = np.array(channel_time_decimated)
            y_decimate = np.array(data_decimated)
            if len(x_decimate) > 1:
                coeffs_deci = np.polyfit(x_decimate, y_decimate, 1)
                ax_time.plot(x_decimate, np.poly1d(coeffs_deci)(x_decimate), 'b--', linewidth=0.7, label="Trendline - Decimate", zorder=30)

            ax_time.plot(channel_time_decimated, data_decimated, label="Decimated Signal", color='red', linestyle='--', rasterized=True, zorder=20)

            # Statistics table
            table_data = [[f"{np.min(data):.6f}", f"{np.median(data):.6f}", f"{np.max(data):.6f}",
                           f"{np.min(data_decimated):.6f}", f"{np.median(data_decimated):.6f}", f"{np.max(data_decimated):.6f}"]]
            col_labels = ["Lower\nORG", "Middle\nORG", "Upper\nORG", "Lower\nDEC", "Middle\nDEC", "Upper\nDEC"]
            the_table = ax_time.table(cellText=table_data, colLabels=col_labels, loc='upper center', cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(4)
            the_table.scale(1, 0.8)
            for _, cell in the_table.get_celld().items():
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
            ax_time.legend(loc="lower right").set_zorder(40)

            # Frequency domain
            dt = np.mean(np.diff(channel_time))
            f_axis = np.fft.rfftfreq(len(data), d=dt)
            magnitude = np.abs(np.fft.rfft(data - mean_original))

            dt_decimated = np.mean(np.diff(channel_time_decimated))
            f_axis_decimated = np.fft.rfftfreq(len(data_decimated), d=dt_decimated)
            magnitude_decimated = np.abs(np.fft.rfft(data_decimated - mean_original))

            ax_freq.plot(f_axis, magnitude, label="FFT Magnitude - ORG", color='blue', zorder=10)
            ax_freq.plot(f_axis_decimated, magnitude_decimated, label="FFT Magnitude - DEC", color='red', zorder=10)
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("Magnitude")
            ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
            ax_freq.grid(True)
            ax_freq.legend(loc="upper right").set_zorder(40)

            # Box plot
            ax_box.boxplot([data, data_decimated], tick_labels=["Original", "Decimated"])
            ax_box.set_title(f"Channel: {channel.name} (Box Plot)")
            ax_box.set_ylabel("Measurement Value")
            ax_box.grid(True)

            # Histogram
            ax_hist.hist(data, bins=30, color='blue', alpha=0.5, label='Original')
            ax_hist.hist(data_decimated, bins=30, color='red', alpha=0.5, label='Decimated')

            mean_original_val = np.mean(data)
            mean_decimated_val = np.mean(data_decimated)
            ax_hist.axvline(mean_original_val, color='blue', linestyle='dashed', linewidth=1.5,
                            label=f'Mean (Original)= {mean_original_val:.2e}' if abs(mean_original_val) < 1e-3 else f'Mean (Original)= {mean_original_val:.2f}')
            ax_hist.axvline(mean_decimated_val, color='red', linestyle='dashed', linewidth=0.75,
                            label=f'Mean (Decimated)= {mean_decimated_val:.2e}' if abs(mean_decimated_val) < 1e-3 else f'Mean (Decimated)= {mean_decimated_val:.2f}')
            ax_hist.set_title(f"Channel: {channel.name} (Histogram)")
            ax_hist.set_xlabel("Measurement Value")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend(fontsize=5)
            ax_hist.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))

        # Turn off unused axes
        for j in range(n_channels, n_rows_fixed):
            for col in range(n_cols):
                axs[j, col].axis("off")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print("Page added.")

    pdf.close()
    print(f"All channel plots saved to {pdf_name}")

