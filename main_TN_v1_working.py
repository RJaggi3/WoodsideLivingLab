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
file_name = "202503281215_SHM-1.tdms"
pdf_name = file_name.replace(".tdms", ".pdf")

# Read the TDMS file and collect all channels
tdms_file = TdmsFile.read(file_name)
all_groups = tdms_file.groups()
channels = []
for group in all_groups:
    for channel in group.channels():
        channels.append(channel)

plt.rcParams.update({'font.size': 7})

# Open a PdfPages object to compile the plots (each figure becomes one A4 page)
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

cnt = 0  
for channel_group in chunks(channels, 6):
    # Create an A4 portrait figure (8.27" x 11.69").
    fig = plt.figure(figsize=(8.27, 11.69))
    plot_num = 321  # Subplot code for a 3x2 grid 
    for channel in channel_group:
        cnt += 1
        # First, extract data from the channel so that `data` is defined
        data = channel[:]  
        # Retrieve the time track; assumes it's available.
        channel_time = channel.time_track()
        
        # Create a subplot using the subplot numbering (321, 322, …, 326)
        ax = plt.subplot(plot_num)
        
        # Plot the profile
        ax.plot(channel_time, data, linewidth=0.75, label="Original Data",rasterized=True,zorder=10)
        
        # Compute and plot the trendline for original data
        x = np.array(channel_time)
        y = np.array(data)
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            trend = p(x)
            # Plot the trendline as a yellow line
            ax.plot(x, trend,  'y-', linewidth=2.0, label="Trendline - Original", zorder=30)
        
        
        
        q= 10 # Decimation factor, changeas needed
        mean_original = np.mean(data)
        data_demeaned = data - mean_original
        data_decimated = decimate(data_demeaned, q, ftype='iir', zero_phase=True)
        data_decimated += mean_original # Add the mean back

        channel_time_decimated = channel_time[::q]

        ax.plot(channel_time_decimated, data_decimated, label="Decimated Signal", color='red', linestyle='--',rasterized=True,zorder=20)
        
        #trendline for decimated data
        x_decimate = np.array(channel_time_decimated)
        y_decimate= np.array(data_decimated)
        if len(x_decimate) > 1:
            coeffs_deci = np.polyfit(x_decimate, y_decimate, 1)
            p_deci = np.poly1d(coeffs_deci)
            trend_decimate = p_deci(x_decimate)
            # Plot the trendline as a blue dash line
            ax.plot(x_decimate, trend_decimate, 'b--', linewidth=0.7, label="Trendline - Decimate", zorder=40)
        
        # Compute and add the table with statistics
        bound_lower = np.min(data)
        bound_middle = np.median(data)
        bound_upper = np.max(data)
        bound_lower_decimated = np.min(data_decimated)
        bound_middle_decimated = np.median(data_decimated)
        bound_upper_decimated = np.max(data_decimated)

        table_data = [[f"{bound_lower:.6f}", f"{bound_middle:.6f}", f"{bound_upper:.6f}",f"{bound_lower_decimated:.6f}", f"{bound_middle_decimated:.6f}", f"{bound_upper_decimated:.6f}"]]
        col_labels = ["Lower\nORG", "Middle\nORG", "Upper\nORG","Lower\nDEC", "Middle\nDEC", "Upper\nDEC"]
        # Add table to the subplot (displayed in the upper center)
        the_table = ax.table(cellText=table_data,
                             colLabels=col_labels,
                             loc='upper center',
                             cellLoc='center')
                             #bbox=[0.5, 0.8, 0.5, 0.2])
                             #colWidths=[0.08, 0.08, 0.08])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(4)
        the_table.scale(1, 0.8)
        for key, cell in the_table.get_celld().items():
            cell.set_facecolor('white') 
            cell.set_alpha(0.8)
            cell.set_edgecolor('black')
            cell.get_text().set_color('black')
        the_table.set_zorder(40)

        # Table properties
        ax.set_xlabel("Time")
        if "unit_string" in channel.properties:
            y_label = channel.properties["unit_string"]
        else:
            y_label = "Measurement Value"
        ax.set_ylabel(y_label)
        ax.set_title(f"Channel: {channel.name}")
        ax.grid(True)
        legend = ax.legend(loc="lower right")
        legend.set_zorder(40)
        plot_num += 1

    # Turn off any unused subplots on this page (if there are fewer than 6 channels)
    total_subplots = 6  # for a 3x2 grid
    num_used = len(channel_group)
    if num_used < total_subplots:
        for j in range(num_used + 1, total_subplots + 1):
            plt.subplot(320 + j).axis("off")

    plt.tight_layout()
    # Save the current figure (one A4 page with 6 or fewer subplots) into the PDF
    pdf.savefig(fig, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Page with channels {cnt - len(channel_group) + 1} to {cnt} added.")

pdf.close()
print(f"All channel plots have been compiled into the file: {pdf_name}")