import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Define your TDMS file name and output PDF name
file_name = "202408070350_SHM-7.tdms"
pdf_name = file_name.replace(".tdms", "_simple_FFT.pdf")

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

# Define a fixed grid of 6 rows x 2 columns (Time Domain and Frequency Domain)
n_rows_fixed = 6
n_cols = 2

# Process channels in groups (up to 6 per page)
for channel_group in chunks(channels, 6):
    n_channels = len(channel_group)
    # Create a new figure with a fixed grid (A3 landscape size: 16.54" x 11.69")
    fig, axs = plt.subplots(nrows=n_rows_fixed, ncols=n_cols, figsize=(16.54, 11.69))
    if n_rows_fixed == 1:
        axs = np.array([axs])
    
    # Loop over each channel in the group
    for i, channel in enumerate(channel_group):
        # Assign axes: Column 0 for Time Domain, Column 1 for Frequency Domain
        ax_time = axs[i, 0]
        ax_freq = axs[i, 1]
        
        # --- TIME DOMAIN PLOT (Raw Data) ---
        data = np.array(channel[:])  # Raw TDMS data
        channel_time = np.array(channel.time_track())  # Corresponding time track
        
        # Plot raw time-domain data
        ax_time.plot(channel_time, data,
                     linewidth=0.75,
                     label="Raw Data",
                     rasterized=True,
                     zorder=10)
        
        # Compute and plot a trendline for the raw data
        if len(channel_time) > 1:
            coeffs = np.polyfit(channel_time, data, 1)
            p = np.poly1d(coeffs)
            trend = p(channel_time)
            ax_time.plot(channel_time, trend,
                         'y-', linewidth=2.0,
                         label="Trendline",
                         zorder=30)
        
        ax_time.set_xlabel("Time")
        y_label = channel.properties["unit_string"] if "unit_string" in channel.properties else "Measurement Value"
        ax_time.set_ylabel(y_label)
        ax_time.set_title(f"Channel: {channel.name} (Time Domain)")
        ax_time.grid(True)
        leg = ax_time.legend(loc="lower right")
        leg.set_zorder(40)
        
        # --- FREQUENCY DOMAIN PLOT (FFT of Raw Data) ---
        dt = np.mean(np.diff(channel_time))  # Estimated sampling interval
        f_axis = np.fft.rfftfreq(len(data), d=dt)
        mean_val = np.mean(data)
        fft_vals = np.fft.rfft(data - mean_val)
        magnitude = np.abs(fft_vals)
        
        ax_freq.plot(f_axis, magnitude,
                     label="FFT Magnitude",
                     color='blue',
                     zorder=10)
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_title(f"Channel: {channel.name} (Frequency Domain)")
        ax_freq.grid(True)
        leg_freq = ax_freq.legend(loc="upper right")
        leg_freq.set_zorder(40)
    
    # Turn off unused axes for any rows that don't contain channel data on this page.
    for j in range(n_channels, n_rows_fixed):
        for col in range(n_cols):
            axs[j, col].axis("off")
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("Page added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")