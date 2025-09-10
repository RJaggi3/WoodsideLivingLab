import pandas as pd
import numpy as np
from scipy import signal
import PyOMA as OMA

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: None

# -------------------------------------------------------------------
# 0. Load & detrend
# -------------------------------------------------------------------
df     = pd.read_csv("acceleration_timeseries.txt", delim_whitespace=True)
t      = df["time_s"].values                   # (n_steps,)
acc    = df.filter(regex="_acc_").values       # (n_steps, n_dof)
acc_dt = signal.detrend(acc, axis=0)           # detrended data

# estimate sampling rate
dt_arr = np.diff(t)
dt     = dt_arr[0] if np.allclose(dt_arr, dt_arr[0]) else np.mean(dt_arr)
fs     = 1.0 / dt

# modal analysis setup
guess_freqs = [0.89, 2.60, 4.10, 5.26, 6.00]
n_modes     = len(guess_freqs)
q_list      = [1, 2, 4, 6, 8]
methods     = ["decimate", "resample", "resample_poly"]
n_methods   = len(methods)

# preallocate storage
freqs        = np.zeros((n_methods, len(q_list), n_modes))
zetas        = np.zeros((n_methods, len(q_list), n_modes))
macs         = np.zeros((n_methods, len(q_list), n_modes))
phi_norm_all = np.zeros((n_methods, len(q_list), n_modes, acc_dt.shape[1]))
phi_ref_norm = np.zeros((n_modes, acc_dt.shape[1]))

# elevations for DOFs
zc       = np.array([0, 5, 10, 16, 20, 25])
z_floors = zc[1:]

# -------------------------------------------------------------------
# 1. Loop over methods and decimation factors
# -------------------------------------------------------------------
for im, method in enumerate(methods):
    for iq, q in enumerate(q_list):
        # downsample and adjust fs_q
        if method == "decimate":
            if q == 1:
                Y, fs_q = acc_dt.copy(), fs
            else:
                Y = signal.decimate(acc_dt, q, axis=0, ftype='iir', zero_phase=True)
                fs_q = fs / q

        elif method == "resample":
            if q == 1:
                Y, fs_q = acc_dt.copy(), fs
            else:
                new_n = int(np.ceil(acc_dt.shape[0] / q))
                Y = signal.resample(acc_dt, new_n, axis=0)
                fs_q = fs / q

        elif method == "resample_poly":
            if q == 1:
                Y, fs_q = acc_dt.copy(), fs
            else:
                Y = signal.resample_poly(acc_dt, up=1, down=q, axis=0)
                fs_q = fs / q

        # SSI covariance-driven stabilization
        SSI_cov = OMA.SSIcovStaDiag(Y, fs_q, br=50)
        plt.close(SSI_cov[0])
        stab    = SSI_cov[1]

        # Mode extraction
        results = OMA.SSIModEX(guess_freqs, stab)
        f_q     = results["Frequencies"]
        z_q     = results["Damping"]
        phi_q   = results["Mode Shapes"]

        # store results
        freqs[im, iq, :] = f_q
        zetas[im, iq, :] = z_q

        # normalize and store mode shapes; compute MAC
        for m in range(n_modes):
            φ_inf = phi_q[m] / np.max(np.abs(phi_q[m]))
            phi_norm_all[im, iq, m, :] = φ_inf

            if q == 1 and method == "decimate":
                phi_ref_norm[m] = φ_inf

            macs[im, iq, m] = OMA.MaC(phi_ref_norm[m], φ_inf)

# -------------------------------------------------------------------
# 2. Plot results into a 5×3 PDF
# -------------------------------------------------------------------
pdf_path = "5_DOF_SSI_methods.pdf"
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(n_modes, 3, figsize=(15, 18), sharex='col')

    for m in range(n_modes):
        # Column 1: Frequency vs q
        ax1 = axes[m, 0]
        for im, method in enumerate(methods):
            ax1.plot(q_list, freqs[im, :, m], '-o', label=method)
        ax1.axhline(freqs[0,0,m], color='k', ls='--', label='Ref')
        if m == 0:
            ax1.set_title("Natural Frequency (Hz)")
        ax1.set_ylabel(f"Mode {m+1}")
        ax1.set_xticks(q_list)
        ax1.legend(fontsize='small')

        # Column 2: Damping vs q
        ax2 = axes[m, 1]
        for im, method in enumerate(methods):
            ax2.plot(q_list, zetas[im, :, m], '-o', label=method)
        ax2.axhline(zetas[0,0,m], color='k', ls='--', label='Ref')
        if m == 0:
            ax2.set_title("Damping Ratio (ζ)")
        ax2.set_xticks(q_list)
        ax2.legend(fontsize='small')

        # Column 3: MAC vs q
        ax3 = axes[m, 2]
        for im, method in enumerate(methods):
            ax3.plot(q_list, macs[im, :, m], '-o', label=method)
        ax3.axhline(1.0, color='k', ls='--', label='Perfect MAC')
        if m == 0:
            ax3.set_title("MAC")
        ax3.set_xlabel("q")
        ax3.legend(fontsize='small')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(f"Saved comparison PDF → {pdf_path}")

# -------------------------------------------------------------------
# 3. Export SSI results to CSV
# -------------------------------------------------------------------
records = []
for im, method in enumerate(methods):
    for iq, q in enumerate(q_list):
        for m in range(n_modes):
            records.append({
                "Method": method,
                "q": q,
                "Mode": m+1,
                "Frequency": freqs[im, iq, m],
                "Damping": zetas[im, iq, m],
                "MAC": macs[im, iq, m]
            })
results_df = pd.DataFrame(records)
results_df.to_csv("ssi_results.csv", index=False)

print("Saved SSI results to 'ssi_results.csv'")