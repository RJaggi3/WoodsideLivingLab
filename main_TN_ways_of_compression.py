import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from nptdms import TdmsFile
from scipy.signal import decimate, resample, resample_poly

# ------------------------------------------------
# Helper function: Compute error metrics
# ------------------------------------------------
def compute_metrics(original, reconstructed, eps=1e-10):
    """
    Compute several error metrics between the original and reconstructed signals.
    Returns MSE, RMSE, SNR (dB), and LSD (log-spectral distance).
    """
    error = original - reconstructed
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    noise_power = np.sum(error ** 2)
    signal_power = np.sum(original ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    # Compute FFT-based log-spectral distance on demeaned signals
    mag_orig = np.abs(np.fft.rfft(original - np.mean(original)))
    mag_recon = np.abs(np.fft.rfft(reconstructed - np.mean(reconstructed)))
    log_diff = np.log(mag_orig + eps) - np.log(mag_recon + eps)
    lsd = np.sqrt(np.mean(log_diff ** 2))
    return mse, rmse, snr, lsd

# ------------------------------------------------
# List of TDMS files to process
# ------------------------------------------------
file_names = ["202503281215_SHM-6.tdms", "202109220920_SHM-6.tdms"]

# ------------------------------------------------
# Define compression factors and methods to compare
# ------------------------------------------------
compression_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
methods = ["Decimate", "Resample", "Resample_poly"]

# Container to store metrics for each file
aggregated_results = {}

# ------------------------------------------------
# Process each file: Compute metrics across all compression factors and methods
# ------------------------------------------------
for file_name in file_names:
    print(f"Processing file: {file_name}")
    tdms_file = TdmsFile.read(file_name)

    # Look for the channel "DYN1-6X"
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

    # Extract time and data from the target channel
    time_data = target_channel.time_track()
    original_data = target_channel[:]

    # Dictionary to store metrics for this file
    metrics_results = {method: {"MSE": [], "RMSE": [], "SNR": [], "LSD": []}
                       for method in methods}

    # Loop over each compression method and factor
    for method in methods:
        for factor in compression_factors:
            if method == "Decimate":
                # Downsample using decimation with anti-alias filtering;
                # then reconstruct using linear interpolation.
                compressed = decimate(original_data, factor,
                                      ftype='iir', zero_phase=True)
                time_compressed = time_data[::factor]
                reconstructed = np.interp(time_data,
                                          time_compressed,
                                          compressed)

            elif method == "Resample":
                # FFT-based resampling: downsample and then upsample back.
                new_length = max(1, len(original_data) // factor)
                compressed = resample(original_data, new_length)
                reconstructed = resample(compressed, len(original_data))

            elif method == "Resample_poly":
                # Polyphase resampling.
                compressed = resample_poly(original_data, up=1, down=factor)
                reconstructed = resample_poly(compressed, up=factor, down=1)
                # Adjust reconstructed length if necessary.
                if len(reconstructed) > len(original_data):
                    reconstructed = reconstructed[:len(original_data)]
                elif len(reconstructed) < len(original_data):
                    reconstructed = np.pad(
                        reconstructed,
                        (0, len(original_data) - len(reconstructed)),
                        mode='edge'
                    )
            else:
                raise ValueError("Unknown compression method encountered.")

            # Compute and store metrics
            mse, rmse, snr, lsd = compute_metrics(original_data,
                                                  reconstructed)
            metrics_results[method]["MSE"].append(mse)
            metrics_results[method]["RMSE"].append(rmse)
            metrics_results[method]["SNR"].append(snr)
            metrics_results[method]["LSD"].append(lsd)

    aggregated_results[file_name] = metrics_results

# ------------------------------------------------
# Create an aggregated figure with a 2x2 grid of subplots
# ------------------------------------------------
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

# Add extra space between rows and columns
fig.subplots_adjust(hspace=0.5, wspace=0.3)

metric_names = ["MSE", "RMSE", "SNR", "LSD"]
colors = {"Decimate": "blue",
          "Resample": "green",
          "Resample_poly": "red"}
line_styles = {
    "202503281215_SHM-6.tdms": "solid",
    "202109220920_SHM-6.tdms": "dashed"
}

# Plot each metric in its subplot
for i, metric in enumerate(metric_names):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for file_name, file_metrics in aggregated_results.items():
        # Assign display labels
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
                marker='o',
                linestyle=line_styles[file_name],
                color=colors[method],
                label=f"{method} ({display_label})"
            )

    ax.set_xlabel("Compression Factor")
    ax.set_xscale("log")
    ax.set_title(metric)
    ax.grid(True)

# ------------------------------------------------
# Create a single global legend outside the subplots
# ------------------------------------------------
all_handles = []
all_labels = []
for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l not in all_labels:
            all_handles.append(h)
            all_labels.append(l)

fig.legend(all_handles,
           all_labels,
           loc='center left',
           bbox_to_anchor=(1.02, 0.5),
           fontsize=8)

fig.suptitle("Combined Error Metrics vs. Compression Factor\n"
             "(Overlaid for both TDMS files)",
             fontsize=18)

# Tight layout, leaving room for the legend
fig.tight_layout(rect=[0, 0, 0.85, 1])

# ------------------------------------------------
# Save the aggregated figure to a PDF
# ------------------------------------------------
pdf_name = "SHM6_error_analysis_combined.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
pdf.savefig(fig)
pdf.close()

plt.close(fig)
print(f"Combined error metrics plot saved as {pdf_name}")
