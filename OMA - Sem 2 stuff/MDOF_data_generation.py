#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic MDOF Time‐History Generator for Woodside‐Style Building
Generates both displacement and acceleration time series at 20 corner‐nodes (X/Y)
using a direct ODE integrator (solve_ivp) instead of Newmark‐β.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh, block_diag
from scipy.integrate import solve_ivp
from pathlib import Path

# 1. MODEL GEOMETRY & DOF MAPPING
zc = np.array([0, 5, 10, 16, 20, 25])    # floor elevations
n_floors = len(zc) - 1                  # 5 above‐ground
n_nodes_per_floor = 4                   # four corners
n_sensors = n_floors * n_nodes_per_floor  # 20 total
ndofs = n_sensors * 2                   # X & Y per node

# Channel labels
channels = []
for s in range(n_sensors):
    channels += [f"DYN1-{s+1}X", f"DYN1-{s+1}Y"]

# 2. MASS MATRIX (M)
x_span, y_span = 12.0, 25.0
slab_area = (2*x_span)*(2*y_span)       # 24 m × 50 m
thickness, density = 0.20, 2500.0        # m, kg/m³
floor_mass = slab_area * thickness * density
mass_per_node = floor_mass / n_nodes_per_floor
M = np.eye(ndofs) * mass_per_node

# 3. STIFFNESS MATRIX (K)
f1 = 1.0                                 # target fundamental freq (Hz)
omega1 = 2*np.pi*f1
M_total = n_floors * floor_mass
K_story = (M_total * omega1**2) / n_floors

# 1D shear‐frame stiffness
K_1d = np.zeros((n_floors, n_floors))
for i in range(n_floors):
    if i > 0:
        K_1d[i, i-1] = K_1d[i-1, i] = -K_story
    K_1d[i, i] = (1 + (i > 0)) * K_story

# replicate for X & Y directions
Kx = np.kron(K_1d, np.eye(n_nodes_per_floor))
Ky = np.kron(K_1d, np.eye(n_nodes_per_floor))
K = block_diag(Kx, Ky)

# 4. DAMPING MATRIX (C) via Rayleigh
vals, _ = eigh(K, M)
omega_n = np.sqrt(vals)
ωa, ωb = omega_n[0], omega_n[2]          # pick modes 1 & 3
ζ_target = 0.05
A_mat = np.array([[1/(2*ωa), ωa/2],
                  [1/(2*ωb), ωb/2]])
b_vec = np.array([ζ_target, ζ_target])
alpha, beta = np.linalg.solve(A_mat, b_vec)
C = alpha * M + beta * K

# 5. EXCITATION & TIME VECTOR
fs, T = 100.0, 200.0                     # Hz, seconds
dt = 1.0 / fs
t = np.arange(0, T, dt)
nt = len(t)

# white‐noise base acceleration
np.random.seed(42)
a_base = 0.1 * np.random.randn(nt)       # std = 0.1 m/s²

# equivalent nodal forces F_time (nt × ndofs)
# F(t) = –M·1·a_base(t), assume uniform base input on all DOFs
ones_vec = np.ones(ndofs)
F_time = -np.outer(a_base, M.dot(ones_vec))

# 6. DEFINE MDOF ODE FOR solve_ivp
M_inv = np.linalg.inv(M)

def mdof_ode(ti, yi):
    # state yi = [u (ndofs), v (ndofs)]
    u = yi[:ndofs]
    v = yi[ndofs:]
    # find nearest time index
    idx = int(np.floor(ti / dt))
    if idx < 0:
        Fi = F_time[0]
    elif idx >= nt:
        Fi = F_time[-1]
    else:
        Fi = F_time[idx]
    du = v
    dv = M_inv.dot(Fi - C.dot(v) - K.dot(u))
    return np.concatenate([du, dv])

# 7. INTEGRATE WITH solve_ivp
y0 = np.zeros(2 * ndofs)  # zero initial displacements & velocities
sol = solve_ivp(mdof_ode,
                (t[0], t[-1]),
                y0,
                t_eval=t,
                method='RK45',
                vectorized=False,
                max_step=dt)

# 8. EXTRACT U, V, COMPUTE A
U = sol.y[:ndofs, :].T        # (nt × ndofs)
V = sol.y[ndofs:, :].T        # (nt × ndofs)

# accelerations by residual: a = M⁻¹ [F - C v - K u]
A_hist = np.zeros_like(U)
for i in range(nt):
    A_hist[i] = M_inv.dot(F_time[i] - C.dot(V[i]) - K.dot(U[i]))

# 9. SAVE TO CSV (same filenames)
df_disp = pd.DataFrame(U, index=t, columns=channels)
df_disp.index.name = "time_s"
df_disp.to_csv(Path("synthetic_disps.csv"))

df_acc = pd.DataFrame(A_hist, index=t, columns=channels)
df_acc.index.name = "time_s"
df_acc.to_csv(Path("synthetic_accs.csv"))

print("Displacements saved to synthetic_disps.csv")
print("Accelerations saved to synthetic_accs.csv")