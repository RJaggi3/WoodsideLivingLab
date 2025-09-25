import numpy as np
import matplotlib.pyplot as plt

from Woodside_simulate import Woodside_sim_gen
from Woodside_downsample_oma import (
    decimate_data,
    resample_data,
    resample_poly_data,
    run_oma,
)

# 1) Generate the simulated 4-DOF acceleration data
data, U, (fn_true, xi_true, phi_true) = Woodside_sim_gen()
fs = 173.61  # original sampling rate [Hz]

# 2) Print the “ground-truth” modal properties
np.set_printoptions(precision=3, suppress=True)
print("True natural frequencies [Hz]:", fn_true)
print("True damping ratio ξ        :", xi_true, "\n")

# 3) Define the down-/resampling q-factors
q_factors = [1, 2, 3, 4]

# 4) Base run (q=1) with no change
print("=== Base run (no decimation or resampling) ===")
base_res = run_oma(
    Y        = data,
    fs       = fs,
    sel_freqs= fn_true.tolist(),
    phi_ref  = phi_true
)
print(f"Estimated freqs [Hz]: {np.round(base_res['Fn'],3)}")
print("MAC matrix vs. true modes:\n", np.round(base_res['MAC'],3), "\n")

# 5) Decimation by q
print("=== Decimation (scipy.signal.decimate) ===")
for q in q_factors:
    Yd  = decimate_data(data, q)
    fsd = fs / q
    res = run_oma(
        Y        = Yd,
        fs       = fsd,
        sel_freqs= fn_true.tolist(),
        phi_ref  = phi_true
    )
    print(f"q={q:>1} → fs={fsd:6.2f} Hz, est freqs: {np.round(res['Fn'],3)}")
    print("MAC:\n", np.round(res['MAC'],3), "\n")
print()

# 6) Fourier‐based resampling (scipy.signal.resample)
print("=== Resample (FFT) down = 1/q ===")
for q in q_factors:
    Yr  = resample_data(data, up=1, down=q)
    fsr = fs / q
    res = run_oma(
        Y        = Yr,
        fs       = fsr,
        sel_freqs= fn_true.tolist(),
        phi_ref  = phi_true
    )
    print(f"1/{q:>1} → fs={fsr:6.2f} Hz, est freqs: {np.round(res['Fn'],3)}")
    print("MAC:\n", np.round(res['MAC'],3), "\n")
print()

# 7) Polyphase resampling (scipy.signal.resample_poly)
print("=== Resample_poly down = 1/q ===")
for q in q_factors:
    Yp  = resample_poly_data(data, up=1, down=q)
    fsp = fs / q
    res = run_oma(
        Y        = Yp,
        fs       = fsp,
        sel_freqs= fn_true.tolist(),
        phi_ref  = phi_true
    )
    print(f"1/{q:>1} → fs={fsp:6.2f} Hz, est freqs: {np.round(res['Fn'],3)}")
    print("MAC:\n", np.round(res['MAC'],3), "\n")
print()