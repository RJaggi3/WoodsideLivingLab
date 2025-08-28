import numpy as np
import pandas as pd
import vibration_toolbox as vtb
from scipy.linalg import eigh
from scipy import signal  # for StateSpace & lsim

# ------------------------------------------------------------------------
# 1. Geometry & Sensor Coordinates (SI units: meters)
# ------------------------------------------------------------------------
zc       = np.array([0, 5, 10, 16, 20, 25])   # elevations (m)
n_dof    = len(zc) - 1                        # 5 DOFs
z_floors = zc[1:]                             # [5, 10, 16, 20, 25]

corner   = "NE"                               # “NE”, “SE”, “SW”, “NW”
x_span, y_span = 12.0, 25.0
corner_map = {
    "NE": ( x_span,  y_span),
    "SE": ( x_span, -y_span),
    "SW": (-x_span, -y_span),
    "NW": (-x_span,  y_span)
}
xy = corner_map[corner]

sensor_coords = np.column_stack([
    np.full(n_dof, xy[0]),   # x (m)
    np.full(n_dof, xy[1]),   # y (m)
    z_floors                 # z (m)
])
print("Sensor coordinates:")
for i, (x, y, z) in enumerate(sensor_coords, start=1):
    print(f"  Floor {i}: x={x:.1f} m, y={y:.1f} m, z={z:.1f} m")
print()

# ------------------------------------------------------------------------
# 2. Lumped mass & story stiffness (unit conversion)
# ------------------------------------------------------------------------
m_ns2_per_mm = 25.91            # N·s² per mm
k_n_per_mm   = 10000            # N per mm

m_floor = m_ns2_per_mm * 1e3    # kg per floor
k_story = k_n_per_mm   * 1e3    # N/m per story

M = np.eye(n_dof) * m_floor
K = np.zeros((n_dof, n_dof))
for i in range(n_dof):
    k_up   = k_story if i < n_dof - 1 else 0.0
    k_down = k_story
    K[i, i] = k_up + k_down
    if i < n_dof - 1:
        K[i, i+1] = -k_story
        K[i+1, i] = -k_story

# ------------------------------------------------------------------------
# 3. Modal damping C so ζ = 2%
# ------------------------------------------------------------------------
eigvals, eigvecs = eigh(K, M)
omega_rad   = np.sqrt(eigvals)    # undamped ω (rad/s)
zeta_target = 0.02                # 2% modal damping

C = M @ eigvecs @ np.diag(2 * zeta_target * omega_rad) @ eigvecs.T @ M

# ------------------------------------------------------------------------
# 4. Modal analysis & print out modes_system results
# ------------------------------------------------------------------------
wn_rad, wd_rad, zeta, Phi, Psi = vtb.mdof.modes_system(M, K, C)

# convert to Hz
wn_hz = wn_rad / (2 * np.pi)
wd_hz = wd_rad / (2 * np.pi)

print("Natural frequencies   (Hz):", np.round(wn_hz, 4))
print("Damped frequencies     (Hz):", np.round(wd_hz, 4))
print("Damping ratios ζ       :", np.round(zeta, 4))

# ∞-norm normalize mode shapes
Phi_inf = Phi / np.max(np.abs(Phi), axis=0)
print("Normalized mode shapes (∞-norm, columns):\n", np.round(Phi_inf, 5))
print()

# ------------------------------------------------------------------------
# 5. Build State‐Space for acceleration outputs
# ------------------------------------------------------------------------
M_inv = np.linalg.inv(M)
zero = np.zeros((n_dof, n_dof))
I_n  = np.eye(n_dof)

A = np.block([
    [ zero,           I_n     ],
    [ -M_inv @ K, -M_inv @ C  ]
])
B = np.vstack([ zero, M_inv ])
C_acc = np.hstack([ -M_inv @ K, -M_inv @ C ])
D_acc = M_inv.copy()

ss_acc = signal.StateSpace(A, B, C_acc, D_acc)

# ------------------------------------------------------------------------
# 6. Time‐domain response via StateSpace + lsim (100 Hz sampling)
# ------------------------------------------------------------------------
dt    = 0.01           # s
t_end = 60.0
t     = np.arange(0, t_end + dt, dt)

rng    = np.random.default_rng(42)
F_time = rng.standard_normal((n_dof, t.size)) * 1e6

t_out, acc_out, x_out = signal.lsim(
    ss_acc,
    U=F_time.T,
    T=t,
    X0=np.zeros(2 * n_dof)
)

a = acc_out  # shape = (n_steps, n_dof)

# ------------------------------------------------------------------------
# 7. Add reduced measurement noise (SNR = 5%)
# ------------------------------------------------------------------------
snr       = 0.05
rng_n     = np.random.default_rng(2025)
signal_std = np.std(a, axis=0)
noise      = rng_n.standard_normal(a.shape) * (signal_std[None, :] * snr)
a_noisy    = a + noise

# ------------------------------------------------------------------------
# 8. Export acceleration time series
# ------------------------------------------------------------------------
acc_data = np.column_stack([t_out, a_noisy])
col_names = ["time_s"] + [f"Floor{j+1}_acc_mps2" for j in range(n_dof)]

np.savetxt(
    "acceleration_timeseries.txt",
    acc_data,
    header=" ".join(col_names),
    comments=''
)
df_acc = pd.DataFrame(acc_data, columns=col_names)
df_acc.to_csv("acceleration_timeseries.csv", index=False)

print("Exported time series (StateSpace) with reduced noise (SNR=5%).")