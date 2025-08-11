#Get the SDOF modal properties using SSI and compare after resampling
#Date: 2025-08-06
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import detrend, decimate, resample, resample_poly

# 1) Force a non‐interactive backend and override plt.show()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: None

import PyOMA

# 2) SDOF simulation parameters
m         = 1.0
fn_true   = 5.0        # true natural frequency [Hz]
zeta_true = 0.02       # true damping ratio
k         = (2 * np.pi * fn_true)**2 * m
c         = 2 * m * (2 * np.pi * fn_true) * zeta_true

# 3) Generate time vector & white‐noise excitation
fs    = 200.0
t     = np.arange(0, 300, 1/fs)
np.random.seed(42)
f_exc = 0.1 * np.random.randn(len(t))

# 4) Integrate SDOF ODE
def sdof_ode(ti, yi):
    x, v = yi
    fi   = np.interp(ti, t, f_exc)
    return [v, (fi - c*v - k*x) / m]

sol = solve_ivp(sdof_ode, (t[0], t[-1]), [0, 0], t_eval=t)
x, v = sol.y

# 5) Compute acceleration and assemble two‐channel data
a     = (f_exc - c*v - k*x) / m
accel = np.vstack([a, a]).T       # shape (N, 2)

# 6) Stabilization‐diagram parameters
br = 50

# 7) Define decimation factors and methods
qs      = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
methods = ['decimate', 'resample', 'resample_poly']

print("Starting OMA downsampling analysis...\n")

# 8) Loop over methods & q, build SSIdatStaDiag → SSIModEX, collect Δf and Δζ
results = {'q': qs}
for method in methods:
    print(f"=== Method: {method} ===")
    df_list = []
    dz_list = []

    for q in qs:
        print(f"  Processing q = {q}...", end='', flush=True)

        # a) Downsample acceleration channels
        if method == 'decimate':
            y0      = decimate(accel[:, 0], q, ftype='iir', zero_phase=True)
            y1      = decimate(accel[:, 1], q, ftype='iir', zero_phase=True)
            accel_q = np.vstack([y0, y1]).T
            fs_q    = fs / q

        elif method == 'resample':
            Nq      = int(np.floor(len(accel) / q))
            y0      = resample(accel[:, 0], Nq)
            y1      = resample(accel[:, 1], Nq)
            accel_q = np.vstack([y0, y1]).T
            fs_q    = fs * (Nq / len(accel))

        else:  # resample_poly
            y0      = resample_poly(accel[:, 0], up=1, down=q, window=('kaiser', 8.0))
            y1      = resample_poly(accel[:, 1], up=1, down=q, window=('kaiser', 8.0))
            accel_q = np.vstack([y0, y1]).T
            fs_q    = fs / q

        # b) Optional detrending
        accel_q = detrend(accel_q, axis=0)

        # c) Build stabilization diagram (auto‐closed)
        fig_stab, Results = PyOMA.SSIdatStaDiag(accel_q, fs_q, br)
        plt.close(fig_stab)

        # d) Extract mode if found
        try:
            Results_mod = PyOMA.SSIModEX([fn_true], Results)
            f_est       = Results_mod['Frequencies'][0]
            z_est       = Results_mod['Damping'][0]
            print(f" OK (f_est={f_est:.3f} Hz, ζ_est={z_est:.4f})")
        except ValueError:
            f_est, z_est = np.nan, np.nan
            print(" FAIL (no stable pole)")

        # e) Store errors
        df_list.append(f_est - fn_true)
        dz_list.append(z_est - zeta_true)

    results[f'Δf_{method}'] = df_list
    results[f'Δζ_{method}'] = dz_list
    print()  # blank line after each method

# 9) Build and print results table
results_df = pd.DataFrame(results).set_index('q')
print("Final Decimation vs. OMA Results (Δf and Δζ):")
print(results_df.round(4))

# 10) Plot all three methods together
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

for method in methods:
    ax1.plot(results_df.index, results_df[f'Δf_{method}'],   '-o', label=method)
    ax2.plot(results_df.index, results_df[f'Δζ_{method}'],   '-o', label=method)

ax1.set_xticks(qs)
ax1.set_ylabel('Δf (Hz)')
ax1.set_title('Frequency‐error Δf vs Compression Factor')
ax1.grid(True, ls=':')
ax1.legend()

ax2.set_xticks(qs)
ax2.set_xlabel('Compression Factor ')
ax2.set_ylabel('Δζ')
ax2.set_title('Damping‐ratio error Δζ vs Compression  Factor')
ax2.grid(True, ls=':')
ax2.legend()

plt.tight_layout()
fig.savefig('OMA_Compression_results.png', dpi=300, bbox_inches='tight')
plt.show()