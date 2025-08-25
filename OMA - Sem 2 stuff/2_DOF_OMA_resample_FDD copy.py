# Force a non-interactive backend and suppress all plt.show()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: None

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import detrend, decimate, resample, resample_poly
import PyOMA

# 1. Define 2-DOF modal properties
fn1, fn2 = 4.0, 8.0
zeta      = 0.02

Phi   = np.array([[1,  1],
                  [1, -1]]) / np.sqrt(2)
Omega = np.diag([2*np.pi*fn1, 2*np.pi*fn2])
M     = np.eye(2)
K     = M.dot(Phi).dot(Omega**2).dot(Phi.T)
C     = 2 * M.dot(Phi).dot(np.diag([zeta, zeta])).dot(Omega).dot(Phi.T)

# 2. Generate time vector & white-noise excitation
fs    = 200.0
t     = np.arange(0, 300, 1/fs)
np.random.seed(42)
f_exc = 0.1 * np.random.randn(len(t), 2)

# 3. Integrate 2-DOF ODE
def two_dof_ode(ti, yi):
    x = yi[:2]; v = yi[2:]
    fi = np.array([
        np.interp(ti, t, f_exc[:,0]),
        np.interp(ti, t, f_exc[:,1])
    ])
    a = np.linalg.solve(M, fi - C.dot(v) - K.dot(x))
    return [v[0], v[1], a[0], a[1]]

y0  = np.zeros(4)
sol = solve_ivp(two_dof_ode, (t[0], t[-1]), y0, t_eval=t)
x1, x2, v1, v2 = sol.y

# 4. Compute accelerations and detrend
accel = np.zeros((len(t), 2))
for i in range(len(t)):
    xi = np.array([x1[i], x2[i]])
    vi = np.array([v1[i], v2[i]])
    accel[i] = np.linalg.solve(M, f_exc[i] - C.dot(vi) - K.dot(xi))
accel = detrend(accel, axis=0)

# 5. Full-rate SSI (reference), stabilization plot suppressed
fig_ref, Results_ref = PyOMA.FDDsvp(accel, fs)
plt.close(fig_ref)
Results_mod_ref      = PyOMA.EFDDmodEX([fn1, fn2], Results_ref)

f_ref, z_ref, Phi_ref = (
    Results_mod_ref['Frequencies'],
    Results_mod_ref['Damping'],
    Results_mod_ref['Mode Shapes']
)

print(f"\nReference frequencies = {f_ref}")
print(f"Reference damping     = {z_ref}\n")

# 6. Compression analysis setup
q_factors = [2, 4, 6, 8, 10]
methods   = ['decimate', 'resample', 'resample_poly']
results   = {m: {} for m in methods}

for method in methods:
    for q in q_factors:
        fs_q = fs / q

        # compress accel → accel_q
        if method == 'decimate':
            accel_q = decimate(accel, q, axis=0, ftype='iir', zero_phase=True)
        elif method == 'resample':
            n_smpl = int(len(accel) / q)
            accel_q = resample(accel, n_smpl, axis=0)
        else:  # 'resample_poly'
            accel_q = resample_poly(accel, up=1, down=q, axis=0)

        accel_q = detrend(accel_q, axis=0)

        # SSI on compressed data, suppress diagram
        fig_q, Results_q = PyOMA.FDDsvp(accel_q, fs_q)
        plt.close(fig_q)
        Results_mod_q    = PyOMA.EFDDmodEX([fn1, fn2], Results_q)

        f_q   = Results_mod_q['Frequencies']
        z_q   = Results_mod_q['Damping']
        Phi_q = Results_mod_q['Mode Shapes']

        mac_q = [PyOMA.MaC(Phi_ref[:,i], Phi_q[:,i]) for i in range(2)]

        results[method][q] = {
            'Frequencies': f_q,
            'Damping':     z_q,
            'MAC':         mac_q
        }

# 7. Print raw modal values
print("\nCompression Results Summary:")
for method in methods:
    print(f"\n=== Method: {method} ===")
    for q in q_factors:
        print(f" q = {q}")
        for i in range(2):
            f_val = results[method][q]['Frequencies'][i]
            z_val = results[method][q]['Damping'][i]
            mac   = results[method][q]['MAC'][i]
            print(f"   Mode {i+1}: Frequency = {f_val:.4f} Hz,  Damping = {z_val:.5f},  MAC = {mac:.4f}")

# 8. Plot MAC vs q (with reference)
plt.figure(figsize=(6,4))
for method in methods:
    for mode in [0,1]:
        macs = [results[method][q]['MAC'][mode] for q in q_factors]
        plt.plot(q_factors, macs, '-o', label=f"{method} M{mode+1}")
# horizontal reference line at MAC = 1.0
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Reference MAC')
plt.xlabel("Compression Factor q")
plt.ylabel("MAC")
plt.title("MAC vs Compression Factor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mac_vs_q.png", dpi=300)
plt.show()

# 9. Plot natural frequency vs q for Mode 1 (with reference)
plt.figure(figsize=(6,4))
for method in methods:
    f1s = [results[method][q]['Frequencies'][0] for q in q_factors]
    plt.plot(q_factors, f1s, '-o', label=method)
# horizontal reference line
plt.axhline(f_ref[0], color='gray', linestyle='--', linewidth=1, label='Ref Mode 1')
plt.xlabel("Compression Factor q")
plt.ylabel("Natural Frequency of Mode 1 [Hz]")
plt.title("FDD: Mode 1 Frequency vs Compression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fdd_mode1_freq_vs_q.png", dpi=300)
plt.show()

# 10. Plot natural frequency vs q for Mode 2 (with reference)
plt.figure(figsize=(6,4))
for method in methods:
    f2s = [results[method][q]['Frequencies'][1] for q in q_factors]
    plt.plot(q_factors, f2s, '-o', label=method)
# horizontal reference line
plt.axhline(f_ref[1], color='gray', linestyle='--', linewidth=1, label='Ref Mode 2')
plt.xlabel("Compression Factor q")
plt.ylabel("Natural Frequency of Mode 2 [Hz]")
plt.title("FDD: Mode 2 Frequency vs Compression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fdd_ode2_freq_vs_q.png", dpi=300)
plt.show()

# 11. Plot damping vs q (with references)
plt.figure(figsize=(6,4))
for method in methods:
    d1s = [results[method][q]['Damping'][0] for q in q_factors]
    d2s = [results[method][q]['Damping'][1] for q in q_factors]
    plt.plot(q_factors, d1s, '-o', label=f"{method} Mode 1")
    plt.plot(q_factors, d2s, '-s', label=f"{method} Mode 2")
# horizontal lines for reference damping
plt.axhline(z_ref[0], color='gray', linestyle='--', linewidth=1, label='Ref Mode 1')
plt.axhline(z_ref[1], color='gray', linestyle=':', linewidth=1, label='Ref Mode 2')
plt.xlabel("Compression Factor q")
plt.ylabel("Damping Ratio")
plt.title("FDD: Extracted Damping vs Compression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fdd_damping_vs_q.png", dpi=300)
plt.show()