import numpy as np
from scipy.signal import decimate, resample, resample_poly
from Woodside_simulate import Woodside_sim_gen
from pyoma2.setup.single import SingleSetup
from pyoma2.algorithms.ssi import SSI
from pyoma2.functions.gen import MAC
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import seaborn as sns


def decimate_data(Y: np.ndarray, q: int, axis: int = 0) -> np.ndarray:
    """
    Downsample each channel by integer factor q using scipy.signal.decimate.
    Args:
        Y     : (N×dof) array of time series
        q     : decimation factor
        axis  : axis along which to decimate (default=0, time axis)
    Returns:
        Y_dec : (N//q × dof) decimated time series
    """
    # zero-phase filtering via filtfilt (default) to avoid aliasing
    return decimate(Y, q, axis=axis, zero_phase=True)

def resample_data(Y: np.ndarray, up: int, down: int, axis: int = 0) -> np.ndarray:
    """
    Change sampling rate by up/down factor via Fourier method.
    Args:
        Y     : (N×dof) array of time series
        up    : up-sampling factor
        down  : down-sampling factor
        axis  : axis along which to resample
    Returns:
        Y_rs  : resampled time series of length ceil(N*up/down)
    """
    N = Y.shape[axis]
    N_new = int(np.ceil(N * up / down))
    return resample(Y, N_new, axis=axis)

def resample_poly_data(Y: np.ndarray, up: int, down: int, axis: int = 0) -> np.ndarray:
    """
    Change sampling rate by up/down factor using polyphase filtering.
    Args:
        Y     : (N×dof) array of time series
        up    : up-sampling factor
        down  : down-sampling factor
        axis  : axis along which to resample
    Returns:
        Y_rp  : resampled time series
    """
    return resample_poly(Y, up, down, axis=axis)

def run_oma(Y: np.ndarray,
            fs: float,
            sel_freqs: list,
            phi_ref: np.ndarray = None,
            br: int = 30,
            ordmax: int = 50,
            step: int = 2) -> dict:
    """
    Run SSI-cov on multichannel data Y, pick modes near sel_freqs,
    and optionally compute Modal Assurance Criterion against phi_ref.

    Args:
        Y         : (N×dof) array of acceleration or displacement data
        fs        : sampling frequency of Y [Hz]
        sel_freqs : list of target frequencies [Hz] for mode picking
        phi_ref   : (dof×n_modes) reference mode shapes for MAC (optional)
        br        : block rows for SSI-cov
        ordmax    : maximum model order for SSI-cov
        step      : model order step size

    Returns:
        dict with keys:
          'order_out' : identified model order
          'Fn'        : array of identified natural frequencies [Hz]
          'Xi'        : array of identified damping ratios
          'Phi'       : identified mode-shape matrix (dof×n_modes)
          'MAC'       : MAC matrix between phi_ref and identified Phi (if phi_ref given)
    """
    # 1) setup OMA
    setup = SingleSetup(Y, fs=fs)

    # 2) configure and add SSI-cov
    ssi_cov = SSI(name="SSI_proc", method="cov", br=br, ordmax=ordmax, step=step)
    setup.add_algorithms(ssi_cov)

    # 3) run identification
    setup.run_all()

    # 4) mode picking around sel_freqs
    setup.mpe(ssi_cov.name, sel_freq=sel_freqs, order_in=ordmax//2)

    # 5) extract results
    res = dict(ssi_cov.result)
    phi_est = res['Phi']  # identified mode shapes
    output = {
        'order_out': res['order_out'],
        'Fn'       : res['Fn'],
        'Xi'       : res['Xi'],
        'Phi'      : phi_est
    }

    # 6) compute MAC if a reference phi_ref is provided
    if phi_ref is not None:
        # ensure both reference and estimated have same DOF×modes shape
        # phi_ref: (dof×n_ref), phi_est: (dof×n_est)
        mac_matrix = MAC(phi_ref, phi_est)
        output['MAC'] = mac_matrix

    return output

if __name__ == "__main__":
    # 1) generate simulated data
    Y, U, (fn_true, xi_true, phi_true) = Woodside_sim_gen()
    fs = 173.61

    print("True freqs [Hz]:", fn_true)
    print("True damping ξ:", xi_true)
    print()

    # 2) define decimation/resampling factors
    dec_factors    = [2, 4, 8]
    resample_ratio = [(3, 2), (5, 4)]    # list of (up, down) pairs
    poly_ratio     = [(7, 5), (9, 8)]

    # 3) run OMA on decimated signals
    for q in dec_factors:
        Yd  = decimate_data(Y, q)
        fsd = fs / q
        res = run_oma(Yd, fsd, sel_freqs=fn_true)
        print(f"Decimate by {q}:_est freqs {np.round(res['Fn'],3)} Hz")

    # 4) run OMA on resampled signals (fourier)
    for up, down in resample_ratio:
        Yr  = resample_data(Y, up, down)
        fsr = fs * up / down
        res = run_oma(Yr, fsr, sel_freqs=fn_true)
        print(f"Resample {up}/{down}: est freqs {np.round(res['Fn'],3)} Hz")

    # 5) run OMA on resampled_poly signals (polyphase)
    for up, down in poly_ratio:
        Yp  = resample_poly_data(Y, up, down)
        fsp = fs * up / down
        res = run_oma(Yp, fsp, sel_freqs=fn_true)
        print(f"Resample_poly {up}/{down}: est freqs {np.round(res['Fn'],3)} Hz")

def plot_ssi_results_pdf(freqs: np.ndarray,
                         zetas: np.ndarray,
                         macs: np.ndarray,
                         q_list: list,
                         methods: list,
                         pdf_path: str):
    """
    Plot SSI-derived natural frequencies, damping ratios and MAC vs. down-sampling factor q,
    for multiple methods, and save to a multi-page PDF.

    Parameters
    ----------
    freqs : ndarray, shape (n_methods, n_q, n_modes)
        Identified natural frequencies [Hz].
    zetas : ndarray, shape (n_methods, n_q, n_modes)
        Identified damping ratios.
    macs : ndarray, shape (n_methods, n_q, n_modes)
        Modal Assurance Criterion values.
    q_list : list of int
        The down-sampling factors q (length n_q).
    methods : list of str
        Names of the methods (length n_methods), e.g. ["decimate","resample","resample_poly"].
    pdf_path : str
        Filepath for the output PDF.
    """
    n_methods, n_q, n_modes = freqs.shape

    # reference lines come from method=0, q=0
    ref_freqs = freqs[0, 0, :]
    ref_zetas = zetas[0, 0, :]

    with PdfPages(pdf_path) as pdf:
        # create one figure with n_modes rows and 3 columns
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 4 * n_modes),
                                 sharex='col')
        for m in range(n_modes):
            # Column 1: Natural Frequency vs q
            ax_f = axes[m, 0]
            for im, method in enumerate(methods):
                ax_f.plot(q_list, freqs[im, :, m], '-o', label=method)
            ax_f.axhline(ref_freqs[m], color='k', ls='--', label='Reference')
            if m == 0:
                ax_f.set_title("Natural Frequency [Hz]")
            ax_f.set_ylabel(f"Mode {m+1}")
            ax_f.set_xticks(q_list)
            ax_f.legend(fontsize='small')

            # Column 2: Damping Ratio vs q
            ax_z = axes[m, 1]
            for im, method in enumerate(methods):
                ax_z.plot(q_list, zetas[im, :, m], '-o', label=method)
            ax_z.axhline(ref_zetas[m], color='k', ls='--', label='Reference')
            if m == 0:
                ax_z.set_title("Damping Ratio ζ")
            ax_z.set_xticks(q_list)
            ax_z.legend(fontsize='small')

            # Column 3: MAC vs q
            ax_mac = axes[m, 2]
            for im, method in enumerate(methods):
                ax_mac.plot(q_list, macs[im, :, m], '-o', label=method)
            ax_mac.axhline(1.0, color='k', ls='--', label='Perfect MAC')
            if m == 0:
                ax_mac.set_title("Modal Assurance Criterion")
            ax_mac.set_xlabel("q")
            ax_mac.set_xticks(q_list)
            ax_mac.legend(fontsize='small')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved SSI comparison plots → {pdf_path}")

def plot_ssi_errors(df: pd.DataFrame, pdf_path: str = None):
    """
    Plot avg. frequency error, avg. damping error, and mean MAC vs. down‐sampling factor q
    for each Method in the SSI results DataFrame, and optionally save to a PDF.

    Parameters
    ----------
    df : pd.DataFrame
        SSI results with columns:
          Method, q, Mode, Frequency, Damping, MAC
    pdf_path : str, optional
        Filepath to write the multi‐panel PDF. If None, the figure is shown instead.
    """
    # 1) Reference values: Method='decimate', q=1
    ref = (
        df
        .query("Method == 'decimate' and q == 1")
        .set_index("Mode")[["Frequency", "Damping"]]
        .rename(columns={"Frequency": "Freq_ref", "Damping": "Zeta_ref"})
    )

    # 2) Join back to compute errors
    df2 = df.join(ref, on="Mode")
    df2["FreqErr%"] = (df2["Frequency"] - df2["Freq_ref"]).abs() \
                      / df2["Freq_ref"] * 100
    df2["ZetaErr%"] = (df2["Damping"] - df2["Zeta_ref"]).abs() \
                      / df2["Zeta_ref"] * 100

    # 3) Aggregate: mean error & mean MAC per Method×q
    agg = (
        df2
        .groupby(["Method", "q"])
        .agg(
            FreqErr_mean=("FreqErr%", "mean"),
            ZetaErr_mean=("ZetaErr%", "mean"),
            MAC_mean=("MAC", "mean")
        )
        .reset_index()
    )

    # 4) Plotting
    sns.set(context="talk", style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # a) Avg. Frequency Error (%)
    sns.lineplot(
        data=agg, x="q", y="FreqErr_mean", hue="Method",
        marker="o", ax=axes[0]
    )
    axes[0].set_title("Avg. Frequency Error (%)")
    axes[0].set_xlabel("Downsampling Factor q")
    axes[0].set_ylabel("Freq Error (%)")
    axes[0].legend(title="Method")

    # b) Avg. Damping Ratio Error (%)
    sns.lineplot(
        data=agg, x="q", y="ZetaErr_mean", hue="Method",
        marker="o", ax=axes[1]
    )
    axes[1].set_title("Avg. Damping Ratio Error (%)")
    axes[1].set_xlabel("Downsampling Factor q")
    axes[1].set_ylabel("ζ Error (%)")
    axes[1].legend_.remove()  # one legend is enough

    # c) Avg. Modal Assurance Criterion
    sns.lineplot(
        data=agg, x="q", y="MAC_mean", hue="Method",
        marker="o", ax=axes[2]
    )
    axes[2].set_title("Avg. Modal Assurance Criterion")
    axes[2].set_xlabel("Downsampling Factor q")
    axes[2].set_ylabel("MAC")
    axes[2].set_ylim(0.9, 1.01)
    axes[2].legend_.remove()

    plt.tight_layout()

    # 5) Save or show
    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Saved SSI error summary PDF → {pdf_path}")
    else:
        plt.show()
