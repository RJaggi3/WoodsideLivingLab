import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import os
import pandas as pd
from load import load_all_channels
from extract_sensor_channels import extract_sensor_channels
from sensor_dictionary import sensor_dictionary
from SSI_input import build_input_matrix
from scipy.signal import decimate, resample, resample_poly
from pyoma2.setup.single import SingleSetup
from pyoma2.algorithms.ssi import SSI
from pyoma2.functions.gen import MAC
from matplotlib.backends.backend_pdf import PdfPages

# Create output directory
output_dir = 'oma_sem2/extract_data/results'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("WOODSIDE 40-DOF DOWNSAMPLING ANALYSIS")
print("Using existing workflow functions")
print("="*80)

# Extract data
print("\nLoading and extracting 40-DOF data...")

file = "oma_sem2/extract_data/202503281215_SHM-6.tdms"
shm6Channels = load_all_channels(file)

sensors = sensor_dictionary()
extracted_channels = extract_sensor_channels(shm6Channels, sensors)

print(f"  Total sensors: {len(sensors)}")
print(f"  Extracted data points: {len(extracted_channels['DYN1-2']['X'])}")

data_2d = build_input_matrix(extracted_channels)
print(f"  Data matrix shape: {data_2d.shape} (time × channels)")

# Detrend
data_dt = signal.detrend(data_2d, axis=0)
print("  Data detrended")

# Get sampling frequency
channel1_time = shm6Channels[0]['time']
time_diffs = np.diff(channel1_time)
fs = 1 / np.mean(time_diffs)
print(f"  Sampling frequency: {fs:.2f} Hz")

# Setup and run reference OMA 
print("\nRunning SSI-COV on original data (reference)...")

# Setup using pyOMA2 
Woodside_setup = SingleSetup(data_dt, fs=fs)

# Apply filters
Woodside_setup.filter_data(Wn=0.02, btype='highpass', order=8)
Woodside_setup.filter_data(Wn=40, btype='lowpass', order=8)
print("  Applied filters")

# Run SSI-COV
ssicov_ref = SSI(name="SSIcov_ref", method="cov", br=30, ordmax=50, calc_unc=False, step=2)
Woodside_setup.add_algorithms(ssicov_ref)
Woodside_setup.run_all()
print("  SSI-COV completed")

# Extract modes
selected_freqs = [1.65, 1.80, 3.06]
Woodside_setup.mpe("SSIcov_ref", sel_freq=selected_freqs, order_in=30)
result_ref = dict(ssicov_ref.result)

omega_ref = result_ref['Fn']
zeta_ref = result_ref['Xi']
phi_ref = result_ref['Phi']
n_modes = len(omega_ref)

print(f"\n  Reference Modal Parameters (n = {n_modes} modes):")
for i in range(n_modes):
    print(f"    Mode {i+1}: f = {omega_ref[i]:.4f} Hz, ζ = {zeta_ref[i]:.6f}")

# Downsample
print("\nDownsampling with 3 methods...")

q_factors = [2, 3, 4, 5, 6, 8, 10]
methods = {
    'decimate': lambda data, q: decimate(data, q, axis=0, zero_phase=True),
    'resample': lambda data, q: resample(data, int(np.ceil(data.shape[0] / q)), axis=0),
    'resample_poly': lambda data, q: resample_poly(data, 1, q, axis=0)
}

print(f"  Downsampling factors: {q_factors}")
print(f"  Methods: {list(methods.keys())}")

# Storage for results
results_all = {method: {} for method in methods.keys()}

for method_name, downsample_func in methods.items():
    print(f"\n  {'='*70}")
    print(f"  METHOD: {method_name.upper()}")
    print(f"  {'='*70}")

    results_all[method_name]['q'] = []
    results_all[method_name]['fs'] = []
    results_all[method_name]['omega'] = []
    results_all[method_name]['zeta'] = []
    results_all[method_name]['phi'] = []
    results_all[method_name]['mac'] = []

    for q in q_factors:
        print(f"\n    Processing q = {q}...")

        # Get filtered data from setup
        data_filtered = Woodside_setup.data

        # Downsample
        data_down = downsample_func(data_filtered, q)
        fs_down = fs / q

        print(f"      Sampling rate: {fs_down:.2f} Hz")
        print(f"      Data points: {data_down.shape[0]}")

        # Run OMA on downsampled data
        try:
            setup_down = SingleSetup(data_down, fs=fs_down)

            ssicov_down = SSI(name=f"SSI_{method_name}_q{q}", method="cov",
                            br=30, ordmax=50, calc_unc=False, step=2)
            setup_down.add_algorithms(ssicov_down)
            setup_down.run_all()

            setup_down.mpe(f"SSI_{method_name}_q{q}", sel_freq=selected_freqs, order_in=30)
            result_down = dict(ssicov_down.result)

            # Handle scalar or array results
            fn_result = result_down['Fn']
            if np.isscalar(fn_result):
                n_found = 1 if not np.isnan(fn_result) else 0
            else:
                n_found = len(fn_result)

            # Pad with NaN if fewer modes found
            omega_q = np.full(n_modes, np.nan)
            zeta_q = np.full(n_modes, np.nan)
            phi_q = np.full_like(phi_ref, np.nan, dtype=complex)

            if n_found > 0:
                # Convert scalars to arrays if needed
                fn_arr = np.atleast_1d(result_down['Fn'])
                xi_arr = np.atleast_1d(result_down['Xi'])
                phi_arr = result_down['Phi']
                if phi_arr.ndim == 1:
                    phi_arr = phi_arr.reshape(-1, 1)

                omega_q[:n_found] = fn_arr[:n_found]
                zeta_q[:n_found] = xi_arr[:n_found]
                phi_q[:, :n_found] = phi_arr[:, :n_found]

            # Calculate MAC
            mac_vals = np.full(n_modes, np.nan)
            if n_found > 0 and phi_q.ndim == 2:
                for m in range(min(n_modes, n_found)):
                    try:
                        phi_r = phi_ref[:, m]
                        phi_d = phi_q[:, m]
                        if not np.any(np.isnan(phi_d)):
                            mac_result = MAC(phi_r.reshape(-1, 1), phi_d.reshape(-1, 1))
                            # MAC returns scalar for single mode comparison
                            mac_vals[m] = np.real(mac_result) if np.isscalar(mac_result) else np.real(mac_result[0, 0])
                    except:
                        pass  # Leave as NaN if MAC calculation fails

            results_all[method_name]['q'].append(q)
            results_all[method_name]['fs'].append(fs_down)
            results_all[method_name]['omega'].append(omega_q)
            results_all[method_name]['zeta'].append(zeta_q)
            results_all[method_name]['phi'].append(phi_q)
            results_all[method_name]['mac'].append(mac_vals)

            print(f"       Extracted {n_found} modes")

        except Exception as e:
            import traceback
            print(f"       Error: {e}")
            # Uncomment for detailed traceback:
            # traceback.print_exc()
            results_all[method_name]['q'].append(q)
            results_all[method_name]['fs'].append(fs_down)
            results_all[method_name]['omega'].append(np.full(n_modes, np.nan))
            results_all[method_name]['zeta'].append(np.full(n_modes, np.nan))
            results_all[method_name]['phi'].append(np.full_like(phi_ref, np.nan))
            results_all[method_name]['mac'].append(np.full(n_modes, np.nan))


# Generate plots

print("\n Generating plots...")

# Prepare data for plotting
method_names = list(methods.keys())
q_list = [1] + q_factors
n_methods = len(method_names)
n_q = len(q_list)

freqs_plot = np.zeros((n_methods, n_q, n_modes))
zetas_plot = np.zeros((n_methods, n_q, n_modes))
macs_plot = np.zeros((n_methods, n_q, n_modes))

# Fill in reference (q=1)
for im in range(n_methods):
    freqs_plot[im, 0, :] = omega_ref
    zetas_plot[im, 0, :] = zeta_ref
    macs_plot[im, 0, :] = 1.0

# Fill in downsampled results
for im, method_name in enumerate(method_names):
    for iq, q in enumerate(q_factors):
        idx_q = iq + 1  # +1 because index 0 is reference
        idx_result = results_all[method_name]['q'].index(q)

        freqs_plot[im, idx_q, :] = results_all[method_name]['omega'][idx_result]
        zetas_plot[im, idx_q, :] = results_all[method_name]['zeta'][idx_result]
        macs_plot[im, idx_q, :] = results_all[method_name]['mac'][idx_result]

# Calculate error arrays
freq_errors = np.zeros((n_methods, n_q, n_modes))
zeta_errors = np.zeros((n_methods, n_q, n_modes))
mac_errors = np.zeros((n_methods, n_q, n_modes))

for im in range(n_methods):
    for iq in range(n_q):
        for m in range(n_modes):
            freq_val = freqs_plot[im, iq, m]
            zeta_val = zetas_plot[im, iq, m]
            mac_val = macs_plot[im, iq, m]

            if not np.isnan(freq_val) and omega_ref[m] != 0:
                freq_errors[im, iq, m] = (freq_val - omega_ref[m]) / omega_ref[m] * 100
            else:
                freq_errors[im, iq, m] = np.nan

            if not np.isnan(zeta_val) and zeta_ref[m] != 0:
                zeta_errors[im, iq, m] = (zeta_val - zeta_ref[m]) / zeta_ref[m] * 100
            else:
                zeta_errors[im, iq, m] = np.nan

            if not np.isnan(mac_val):
                mac_errors[im, iq, m] = (1.0 - mac_val) * 100  # % loss from perfect correlation
            else:
                mac_errors[im, iq, m] = np.nan

# Create plots - Modal Parameters
with PdfPages(f'{output_dir}/modal_parameters_vs_q.pdf') as pdf:
    fig, axes = plt.subplots(n_modes, 3, figsize=(15, 4 * n_modes), sharex='col')
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for m in range(n_modes):
        # Frequency
        for im, method_name in enumerate(method_names):
            axes[m, 0].plot(q_list, freqs_plot[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 0].axhline(omega_ref[m], color='k', ls='--', alpha=0.5)
        axes[m, 0].set_ylabel(f'Mode {m+1} Freq (Hz)')
        axes[m, 0].set_title('Natural Frequency' if m == 0 else '')
        axes[m, 0].grid(True, alpha=0.3)
        axes[m, 0].legend()

        # Damping
        for im, method_name in enumerate(method_names):
            axes[m, 1].plot(q_list, zetas_plot[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 1].axhline(zeta_ref[m], color='k', ls='--', alpha=0.5)
        axes[m, 1].set_ylabel(f'Mode {m+1} Damping')
        axes[m, 1].set_title('Damping Ratio ζ' if m == 0 else '')
        axes[m, 1].grid(True, alpha=0.3)

        # MAC
        for im, method_name in enumerate(method_names):
            axes[m, 2].plot(q_list, macs_plot[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 2].axhline(1.0, color='k', ls='--', alpha=0.5)
        axes[m, 2].set_ylabel(f'Mode {m+1} MAC')
        axes[m, 2].set_title('MAC' if m == 0 else '')
        axes[m, 2].set_xlabel('Downsampling factor q')
        axes[m, 2].set_ylim([0, 1.1])
        axes[m, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Saved: modal_parameters_vs_q.pdf")

# Create error plots
with PdfPages(f'{output_dir}/error_analysis.pdf') as pdf:
    fig, axes = plt.subplots(n_modes, 3, figsize=(15, 4 * n_modes), sharex='col')
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for m in range(n_modes):
        # Frequency error
        for im, method_name in enumerate(method_names):
            axes[m, 0].plot(q_list, freq_errors[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 0].axhline(0, color='k', ls='--', alpha=0.5)
        axes[m, 0].set_ylabel(f'Mode {m+1} Δω (%)')
        axes[m, 0].set_title('Frequency Error' if m == 0 else '')
        axes[m, 0].grid(True, alpha=0.3)
        axes[m, 0].legend()

        # Damping error
        for im, method_name in enumerate(method_names):
            axes[m, 1].plot(q_list, zeta_errors[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 1].axhline(0, color='k', ls='--', alpha=0.5)
        axes[m, 1].set_ylabel(f'Mode {m+1} Δζ (%)')
        axes[m, 1].set_title('Damping Error' if m == 0 else '')
        axes[m, 1].grid(True, alpha=0.3)

        # MAC error (loss from perfect)
        for im, method_name in enumerate(method_names):
            axes[m, 2].plot(q_list, mac_errors[im, :, m], '-o', label=method_name, markersize=6)
        axes[m, 2].axhline(0, color='k', ls='--', alpha=0.5)
        axes[m, 2].set_ylabel(f'Mode {m+1} MAC Loss (%)')
        axes[m, 2].set_title('MAC Degradation' if m == 0 else '')
        axes[m, 2].set_xlabel('Downsampling factor q')
        axes[m, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Saved: error_analysis.pdf")

# Save results
print("\n Saving results...")

with open(f'{output_dir}/results.pkl', 'wb') as f:
    pickle.dump({
        'omega_ref': omega_ref,
        'zeta_ref': zeta_ref,
        'phi_ref': phi_ref,
        'results_all': results_all,
        'q_factors': q_factors,
        'fs_original': fs
    }, f)
print(f"  Saved: results.pkl")

# Create summary report
with open(f'{output_dir}/SUMMARY_REPORT.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("WOODSIDE 40-DOF DOWNSAMPLING ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write(f"Dataset: {file}\n")
    f.write(f"Sensors: {len(sensors)}\n")
    f.write(f"Channels: 40 (20 sensors × 2 directions)\n")
    f.write(f"Sampling rate: {fs:.2f} Hz\n\n")

    f.write("Reference Modal Parameters:\n")
    for i in range(n_modes):
        f.write(f"  Mode {i+1}: f = {omega_ref[i]:.4f} Hz, ζ = {zeta_ref[i]:.6f}\n")
    f.write("\n")

    for method_name in method_names:
        f.write(f"\n{'='*70}\n")
        f.write(f"METHOD: {method_name.upper()}\n")
        f.write(f"{'='*70}\n")

        for iq, q in enumerate(q_factors):
            fs_q = fs / q
            f.write(f"\n  q = {q} (fs = {fs_q:.2f} Hz):\n")

            omega_q = results_all[method_name]['omega'][iq]
            zeta_q = results_all[method_name]['zeta'][iq]

            for m in range(n_modes):
                if not np.isnan(omega_q[m]):
                    freq_err = (omega_q[m] - omega_ref[m]) / omega_ref[m] * 100
                    zeta_err = (zeta_q[m] - zeta_ref[m]) / zeta_ref[m] * 100
                    f.write(f"    Mode {m+1}: f = {omega_q[m]:.4f} Hz ({freq_err:+.2f}%), ")
                    f.write(f"ζ = {zeta_q[m]:.6f} ({zeta_err:+.2f}%)\n")
                else:
                    f.write(f"    Mode {m+1}: NOT FOUND\n")

print(f"  Saved: SUMMARY_REPORT.txt")

print("\n" + "="*80)
print("WORKFLOW COMPLETE!")
print("="*80)
print(f"\nAll outputs saved in: {output_dir}/")
print("\nFiles generated:")
print("  - modal_parameters_vs_q.pdf")
print("  - error_analysis.pdf")
print("  - results.pkl")
print("  - SUMMARY_REPORT.txt")
print("="*80)
