"""
5-DOF shear building response with three down-sampling methods:
decimate, resample, resample_poly
- Newmark-beta (average acceleration) time integration
- Rayleigh damping calibrated to 5% at modes 1 and 3
- Plots ground accel, floor displacements, inter-storey drifts, and base shear
- Builds table of peak displacement differences (mm) vs reference

Dependencies: numpy, scipy, matplotlib, pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample, resample_poly
from scipy.linalg import eigh, cho_factor, cho_solve

# ---------------------------
# Plotting function
# ---------------------------
def plot_results(t, ag, u, drifts, Vb, title_suffix=""):
    N = u.shape[1]
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(t, ag/9.80665, lw=1)
    plt.xlabel("Time (s)"); plt.ylabel("Ground accel (g)")
    plt.title(f"Ground accel {title_suffix}")
    plt.subplot(2, 2, 2)
    for i in range(N):
        plt.plot(t, u[:, i]*1000.0, label=f"Floor {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("Displacement (mm)")
    plt.title(f"Floor disp {title_suffix}"); plt.legend()
    plt.subplot(2, 2, 3)
    for i in range(N):
        plt.plot(t, drifts[:, i]*1000.0, label=f"Storey {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("Drift (mm)")
    plt.title(f"Inter-storey drifts {title_suffix}"); plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(t, Vb/1e6, lw=1.2)
    plt.xlabel("Time (s)"); plt.ylabel("Base shear (MN)")
    plt.title(f"Base shear {title_suffix}")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Ground motion loader
# ---------------------------
def load_ground_motion(path="elcentro.dat"):
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None, names=["t", "g"])
        t = df["t"].to_numpy(float)
        ag = 9.80665 * df["g"].to_numpy(float)
        return t, ag
    except:
        t = np.arange(0, 10, 0.02)
        ag = 0.2 * 9.80665 * np.sin(2 * np.pi * 1.0 * t)
        return t, ag

# ---------------------------
# Structural model assembly
# ---------------------------
def shear_building_matrices(masses, story_stiffness):
    m = np.array(masses); k = np.array(story_stiffness)
    N = m.size
    M = np.diag(m); K = np.zeros((N, N))
    for i in range(N):
        if i == 0:
            K[i, i] = k[i] + (k[i+1] if N>1 else 0)
            if N>1:
                K[i, i+1] = K[i+1, i] = -k[i+1]
        elif i == N-1:
            K[i, i] = k[i]
        else:
            K[i, i] = k[i] + k[i+1]
            K[i, i+1] = K[i+1, i] = -k[i+1]
    return M, K

def rayleigh_damping(M, K, zeta=0.05, modes=(0, 2)):
    w2, Phi = eigh(K, M)
    w = np.sqrt(np.clip(w2, 0, None))
    i, j = modes; wi, wj = w[i], w[j]
    A = np.array([[1/(2*wi), wi/2], [1/(2*wj), wj/2]])
    b = np.array([zeta, zeta])
    a0, a1 = np.linalg.solve(A, b)
    C = a0 * M + a1 * K
    return C, (a0, a1), w, Phi

def newmark_average_accel(M, C, K, t, ag, r=None, u0=None, v0=None):
    gamma, beta = 0.5, 0.25
    N = M.shape[0]
    if r is None: r = np.ones(N)
    if u0 is None: u0 = np.zeros(N)
    if v0 is None: v0 = np.zeros(N)
    dt = np.mean(np.diff(t)); n = len(t)
    A_ = (1/(beta*dt*dt))*M + (gamma/(beta*dt))*C
    K_eff = K + A_
    a0 = 1/(beta*dt*dt); a1 = 1/(beta*dt); a2 = 1/(2*beta)-1
    a3 = gamma/(beta*dt); a4 = gamma/beta-1; a5 = dt*(gamma/(2*beta)-1)
    u = np.zeros((n, N)); v = np.zeros((n, N)); a = np.zeros((n, N))
    a[0] = np.linalg.solve(M, -C@v0 - K@u0 - M@(r*ag[0]))
    u[0], v[0] = u0, v0
    def eff_force(u_k, v_k, a_k, ag_k1):
        rhs = -M@(r*ag_k1)
        return rhs + M@(a0*u_k + a1*v_k + a2*a_k) + C@(a3*u_k + a4*v_k + a5*a_k)
    L, low = cho_factor(K_eff, overwrite_a=False, check_finite=True)
    for k_ in range(n-1):
        p = eff_force(u[k_], v[k_], a[k_], ag[k_+1])
        u[k_+1] = cho_solve((L, low), p)
        a[k_+1] = a0*(u[k_+1]-u[k_]) - a1*v[k_] - a2*a[k_]
        v[k_+1] = v[k_] + dt*((1-gamma)*a[k_] + gamma*a[k_+1])
    restoring = (K@u.T).T; damp = (C@v.T).T; inert = (M@a.T).T
    Vb = (restoring + damp + inert).sum(axis=1)
    return u, v, a, Vb

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    t_ref, ag_ref = load_ground_motion()

    N = 5
    masses = np.array([5490991.3687, 10000, 10000, 10000, 32235.6917])
    story_k = np.array([9.6483e8, 8.2428e6, 9.6120e8, 9.8259e8, 2.2017e6])
    M, K = shear_building_matrices(masses, story_k)
    C, _, _, _ = rayleigh_damping(M, K, zeta=0.05, modes=(0,2))

    # Reference case
    u_ref, v_ref, a_ref, Vb_ref = newmark_average_accel(M, C, K, t_ref, ag_ref, r=np.ones(N))
    drifts_ref = np.zeros_like(u_ref)
    drifts_ref[:,0] = u_ref[:,0]
    for i in range(1, N):
        drifts_ref[:,i] = u_ref[:,i] - u_ref[:,i-1]
    peak_ref = np.max(np.abs(u_ref))
    print(f"Reference peak displacement: {peak_ref*1000:.2f} mm")
    plot_results(t_ref, ag_ref, u_ref, drifts_ref, Vb_ref, "(ref)")

    # Prepare table of differences
    methods = ["decimate", "resample", "resample_poly"]
    qs = [2,3,4,5,6,7,8]
    diff_table = pd.DataFrame(index=methods, columns=qs)

    # Down-sampling methods
    for q in qs:
        # decimate
        ag_d = decimate(ag_ref, q, ftype='iir'); t_d = t_ref[::q]
        u_d, _, _, Vb_d = newmark_average_accel(M, C, K, t_d, ag_d, r=np.ones(N))
        drifts_d = np.zeros_like(u_d); drifts_d[:,0] = u_d[:,0]
        for i in range(1, N):
            drifts_d[:,i] = u_d[:,i] - u_d[:,i-1]
        peak_d = np.max(np.abs(u_d))
        diff_table.at["decimate", q] = abs(peak_d - peak_ref) * 1000
        plot_results(t_d, ag_d, u_d, drifts_d, Vb_d, f"(decimate q={q})")

        # resample
        Nt = int(len(ag_ref) / q)
        ag_rs = resample(ag_ref, Nt); t_rs = np.linspace(t_ref[0], t_ref[-1], Nt)
        u_rs, _, _, Vb_rs = newmark_average_accel(M, C, K, t_rs, ag_rs, r=np.ones(N))
        drifts_rs = np.zeros_like(u_rs); drifts_rs[:,0] = u_rs[:,0]
        for i in range(1, N):
            drifts_rs[:,i] = u_rs[:,i] - u_rs[:,i-1]
        peak_rs = np.max(np.abs(u_rs))
        diff_table.at["resample", q] = abs(peak_rs - peak_ref) * 1000
        plot_results(t_rs, ag_rs, u_rs, drifts_rs, Vb_rs, f"(resample q={q})")

        # resample_poly
        ag_rp = resample_poly(ag_ref, up=1, down=q); t_rp = t_ref[::q]
        u_rp, _, _, Vb_rp = newmark_average_accel(M, C, K, t_rp, ag_rp, r=np.ones(N))
        drifts_rp = np.zeros_like(u_rp); drifts_rp[:,0] = u_rp[:,0]
        for i in range(1, N):
            drifts_rp[:,i] = u_rp[:,i] - u_rp[:,i-1]
        peak_rp = np.max(np.abs(u_rp))
        diff_table.at["resample_poly", q] = abs(peak_rp - peak_ref) * 1000
        plot_results(t_rp, ag_rp, u_rp, drifts_rp, Vb_rp, f"(resample_poly q={q})")

    # Display the difference table
    print("\nPeak displacement differences (mm):")
    print(diff_table)
