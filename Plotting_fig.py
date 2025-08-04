import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate, resample, resample_poly

###############################################################################
# PART 1: Compute Error Metrics and Plot (2x2 grid with global legend)
###############################################################################

def compute_metrics(original, reconstructed, eps=1e-10):
    """
    Compute error metrics between original and reconstructed signals.
    Returns MSE, RMSE, SNR (dB) and LSD.
    """
    error = original - reconstructed
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    noise_power = np.sum(error**2)
    signal_power = np.sum(original**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    mag_orig = np.abs(np.fft.rfft(original - np.mean(original)))
    mag_recon = np.abs(np.fft.rfft(reconstructed - np.mean(reconstructed)))
    log_diff = np.log(mag_orig + eps) - np.log(mag_recon + eps)
    lsd = np.sqrt(np.mean(log_diff**2))
    return mse, rmse, snr, lsd

# List of TDMS file names.
file_names = ["202503281215_SHM-6.tdms", "202109220920_SHM-6.tdms"]

# Define compression factors and the methods to test.
compression_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
methods = ["Decimate", "Resample", "Resample_poly"]

# Container for aggregated error metrics (per file).
aggregated_results = {}

for file_name in file_names:
    print(f"Processing file: {file_name}")
    tdms_file = TdmsFile.read(file_name)
    
    # Select channel "DYN1-6X"
    target_channel = None
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name == "DYN1-6X":
                target_channel = channel
                break
        if target_channel:
            break
    if target_channel is None:
        raise ValueError(f"Channel 'DYN1-6X' not found in file {file_name}.")

    time_data = target_channel.time_track()
    original_data = target_channel[:]
    
    metrics_results = {method: {"MSE": [], "RMSE": [], "SNR": [], "LSD": []}
                       for method in methods}
    
    for method in methods:
        for factor in compression_factors:
            if method == "Decimate":
                # Downsample with anti-alias filtering then reconstruct using linear interpolation.
                compressed = decimate(original_data, factor, ftype="iir", zero_phase=True)
                time_compressed = time_data[::factor]
                reconstructed = np.interp(time_data, time_compressed, compressed)
            elif method == "Resample":
                new_length = max(1, len(original_data) // factor)
                compressed = resample(original_data, new_length)
                reconstructed = resample(compressed, len(original_data))
            elif method == "Resample_poly":
                compressed = resample_poly(original_data, up=1, down=factor)
                reconstructed = resample_poly(compressed, up=factor, down=1)
                if len(reconstructed) > len(original_data):
                    reconstructed = reconstructed[:len(original_data)]
                elif len(reconstructed) < len(original_data):
                    reconstructed = np.pad(reconstructed, 
                                           (0, len(original_data) - len(reconstructed)),
                                           mode="edge")
            else:
                raise ValueError("Unknown compression method encountered.")
                
            mse, rmse, snr, lsd = compute_metrics(original_data, reconstructed)
            metrics_results[method]["MSE"].append(mse)
            metrics_results[method]["RMSE"].append(rmse)
            metrics_results[method]["SNR"].append(snr)
            metrics_results[method]["LSD"].append(lsd)
            
    aggregated_results[file_name] = metrics_results

# Create a 2x2 grid for the error metrics.
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
metric_names = ["MSE", "RMSE", "SNR", "LSD"]
colors = {"Decimate": "blue", "Resample": "green", "Resample_poly": "red"}
# Define line styles: solid for "202503281215_SHM-6.tdms" 
# and dotted for "202109220920_SHM-6.tdms".
line_styles = {
    "202503281215_SHM-6.tdms": "solid",
    "202109220920_SHM-6.tdms": ":"
}

for i, metric in enumerate(metric_names):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for file_name, file_metrics in aggregated_results.items():
        if file_name == "202503281215_SHM-6.tdms":
            display_label = "Operational State"
        elif file_name == "202109220920_SHM-6.tdms":
            display_label = "2021 Mansfield Earthquake"
        else:
            display_label = file_name
        for method in methods:
            ax.plot(
                compression_factors,
                file_metrics[method][metric],
                marker="o",
                linestyle=line_styles[file_name],
                color=colors[method],
                label=f"{method} ({display_label})"
            )
    ax.set_xlabel("Resampling Factor")
    ax.set_xscale("log")
    ax.set_title(metric)
    ax.grid(True)
    
# Build a single (global) legend without modifying default legend symbols.
all_handles = []
all_labels = []
for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l not in all_labels:
            all_handles.append(h)
            all_labels.append(l)
print("Unique legend entries:", all_labels)

plt.subplots_adjust(right=0.70)
fig.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(0.72, 0.5),
           fontsize=8, frameon=True)
#fig.suptitle("Combined Error Metrics vs. Compression Factor\n(Overlaid for both TDMS files)", fontsize=18)
plt.show()

pdf_name = "SHM6_error_analysis_combined.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
pdf.savefig(fig)
pdf.close()
plt.close(fig)
print(f"Combined error metrics plot saved as {pdf_name}")


###############################################################################
# PART 2: Raw Data Time-Series Plot (1x2 grid with same vertical scale and y-axis number labels)
###############################################################################

# Collect raw data and time arrays from each file.
time_series = []
for file_name in file_names:
    tdms_file = TdmsFile.read(file_name)
    target_channel = None
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name == "DYN1-6X":
                target_channel = channel
                break
        if target_channel:
            break
    if target_channel is None:
        raise ValueError(f"Channel 'DYN1-6X' not found in file {file_name}.")
    time_data = target_channel.time_track()
    raw_data = target_channel[:]
    if file_name == "202503281215_SHM-6.tdms":
        label = "Operational State"
    elif file_name == "202109220920_SHM-6.tdms":
        label = "2021 Mansfield Earthquake"
    else:
        label = file_name
    time_series.append((time_data, raw_data, label))

# Determine the common (minimum) sample count across files.
common_length = min(len(raw_data) for (_, raw_data, _) in time_series)
print("Common data length:", common_length)

# Compute global y-axis limits across both signals.
global_ymin = min(np.min(raw_data[:common_length]) for (_, raw_data, _) in time_series)
global_ymax = max(np.max(raw_data[:common_length]) for (_, raw_data, _) in time_series)
print("Global Y limits:", global_ymin, global_ymax)

# Create a 1x2 grid of subplots (horizontal layout) for the raw data time-series.
fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
for i, (time_data, raw_data, label) in enumerate(time_series):
    axs2[i].plot(time_data[:common_length], raw_data[:common_length])
    axs2[i].set_title(label)
    axs2[i].set_xlabel("Time")
    axs2[i].grid(True)
    axs2[i].set_ylim(global_ymin, global_ymax)
    # Ensure that numeric tick labels are visible on the y-axis for both subplots.
    axs2[i].tick_params(axis='y', labelleft=True)
    # Set the y-axis label on each subplot
    axs2[i].set_ylabel("Volts")
    
plt.tight_layout()
plt.show()

raw_pdf_name = "Raw_data_over_time.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(raw_pdf_name)
pdf.savefig(fig2)
pdf.close()
print(f"Raw data plot saved as {raw_pdf_name}")