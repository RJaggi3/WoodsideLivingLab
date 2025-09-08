import pandas as pd
import numpy as np
from scipy import signal
import PyOMA as OMA
import matplotlib.pyplot as plt

# 1. Load and detrend 
df     = pd.read_csv("acceleration_timeseries.txt", delim_whitespace=True)
t      = df["time_s"].values
acc    = df.filter(regex="_acc_").values
acc_dt = signal.detrend(acc, axis=0)

# 2. Estimate sampling freq & run SSI 
dt_arr   = np.diff(t)
dt       = dt_arr[0] if np.allclose(dt_arr, dt_arr[0]) else np.mean(dt_arr)
fs       = 1.0 / dt
FreQ     = [0.89, 2.60, 4.10, 5.26, 6.00]
SSI_cov  = OMA.SSIcovStaDiag(acc_dt, fs, br=15)
results  = OMA.SSIModEX(FreQ, SSI_cov[1])

nat_freqs   = results['Frequencies']
dampings    = results['Damping']
mode_shapes = results['Mode Shapes']

# 3. Reconstruct sensor_coords & elevations
zc       = np.array([0, 5, 10, 16, 20, 25])
n_dof    = len(zc) - 1
z_floors = zc[1:]
corner   = "NE"
x_span, y_span = 12.0, 25.0
corner_map = {
    "NE": ( x_span,  y_span),
    "SE": ( x_span, -y_span),
    "SW": (-x_span, -y_span),
    "NW": (-x_span,  y_span)
}
xy = corner_map[corner]
sensor_coords = np.column_stack([
    np.full(n_dof, xy[0]),
    np.full(n_dof, xy[1]),
    z_floors
])
z_coords = sensor_coords[:, 2]

# 4. Create PDF with 5 rows × 1 column of subplots
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 15), sharex=True)
for i, ax in enumerate(axes):
    φ = mode_shapes[i]
    φ_inf = φ / np.max(np.abs(φ))  # ∞-norm normalize

    ax.plot(z_coords, φ_inf, '-o', lw=2, color=f'C{i}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel("Norm amplitudes")
    ax.set_title(f"Mode {i+1}: {nat_freqs[i]:.2f} Hz, ζ={dampings[i]:.2%}")
    ax.grid(True)

axes[-1].set_xlabel("Elevation (m)")
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save to PDF
pdf_filename = "mode_shapes.pdf"
plt.savefig(pdf_filename)
plt.close()

print(f"Saved mode shapes plot to '{pdf_filename}'")