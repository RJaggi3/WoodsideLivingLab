import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

# ---------------------------
# 1. User Inputs
# ---------------------------

# Three target natural frequencies (Hz)
f_target = np.array([1.62615378, 1.84210426, 3.0479326 ])

# Three target damping ratios (fraction)
zeta_target = np.array([0.10983557, 0.06689684, 0.15441141])

# Full 40×3 mode‐shape matrix (rows = sensors X and Y, columns = modes)
Phi_full = np.array([
 [ 1.75232159e-01,  1.01327509e-01,  3.87016506e-01],
 [-8.75837186e-02,  5.38602117e-01,  3.32896370e-01],
 [ 3.80734545e-05, -2.29354308e-04,  9.81099812e-04],
 [ 1.81052798e-03,  5.01897338e-03, -9.27916930e-04],
 [-3.80829004e-03, -1.41162191e-02,  1.54365933e-02],
 [ 4.61528790e-02,  2.97685387e-02,  1.98939394e-02],
 [ 4.56515921e-02,  8.47371561e-02,  5.07697317e-02],
 [ 7.35882607e-02,  6.99126439e-02,  2.48518614e-02],
 [ 4.41075424e-01,  5.57122910e-02, -5.03817261e-02],
 [-2.15413116e-02,  1.73160162e-01,  6.34570697e-02],
 [ 7.61919380e-02,  4.13812063e-02,  1.51843461e-01],
 [ 7.25388275e-03,  1.70586470e-01,  1.27535017e-02],
 [ 3.30051823e-01,  4.58949650e-02,  3.61581408e-02],
 [ 5.74662299e-02,  2.72984939e-01, -2.28144768e-01],
 [ 1.86758905e-01,  1.20005974e-01,  4.97323472e-01],
 [ 3.41800577e-02,  3.48299642e-01, -1.06805729e-01],
 [ 8.18562446e-01,  1.41496847e-01, -2.15091827e-01],
 [-5.26741935e-02,  8.06975005e-01,  2.25639321e-01],
 [ 1.26589087e-01,  4.38396670e-02,  9.97357291e-02],
 [-8.94116940e-03,  5.52246273e-01,  2.94414144e-03],
 [ 7.14552015e-01,  7.16533171e-02, -1.68127159e-01],
 [ 1.98184944e-01,  6.76403447e-01, -3.12720445e-01],
 [ 2.44276428e-01,  8.32396930e-02,  8.01734874e-01],
 [ 1.83216543e-01,  7.55285171e-01, -2.66516646e-01],
 [ 9.72424970e-01,  1.15458841e-01, -3.25093429e-01],
 [-4.10880244e-04,  9.61052820e-01,  2.41881154e-01],
 [ 1.36772496e-01,  9.54341556e-02,  1.84130078e-01],
 [ 2.00730243e-02,  4.31949804e-01, -4.03303426e-02],
 [ 7.97237237e-01,  7.26267081e-02, -1.98330753e-01],
 [ 3.68930641e-01,  8.66904727e-01, -4.51864598e-01],
 [ 3.75620742e-01,  1.42482415e-01,  1.00000000e+00],
 [ 2.73392448e-01,  9.32427363e-01, -3.13519358e-01],
 [ 1.00000000e+00,  1.93640572e-02, -3.88929321e-01],
 [-8.63968140e-02,  9.36708829e-01,  3.11595999e-01],
 [ 1.47197085e-01,  1.23238785e-01,  3.49245597e-01],
 [-2.96651825e-02,  1.00000000e+00,  1.97904744e-01],
 [ 8.68451810e-01,  1.64476746e-01, -2.36816673e-01],
 [ 2.85120791e-01,  9.09314654e-01, -2.86099903e-01],
 [ 1.06891227e-01,  5.43788024e-02,  1.88836348e-01],
 [-1.17163927e-04,  1.04566081e-03, -2.52187659e-04],
])

# ---------------------------
# 2. Floor-to-sensor mapping (X only, 5 levels LG–3)
# ---------------------------
floor_cables = {
    'LG': ['DYN1-2','DYN1-3','DYN1-4','DYN1-5'],
     'G': ['DYN1-9','DYN1-10','DYN1-11','DYN1-12'],
    '1': ['DYN1-16','DYN1-17','DYN1-18','DYN1-19'],
    '2': ['DYN1-23','DYN1-24','DYN1-25','DYN1-26'],
    '3': ['DYN1-30','DYN1-31','DYN1-32','DYN1-33'],
}

# Map each X-channel ID to its row index in Phi_full (even indices 0–38)
sensor_index = {
    'DYN1-2':  0, 'DYN1-3':  2, 'DYN1-4':  4, 'DYN1-5':  6,
    'DYN1-9':  8, 'DYN1-10':10, 'DYN1-11':12, 'DYN1-12':14,
    'DYN1-16':16,'DYN1-17':18,'DYN1-18':20,'DYN1-19':22,
    'DYN1-23':24,'DYN1-24':26,'DYN1-25':28,'DYN1-26':30,
    'DYN1-30':32,'DYN1-31':34,'DYN1-32':36,'DYN1-33':38,
}

# ---------------------------
# 3. Reduce to 5×3 floor‐averaged mode shapes
# ---------------------------
Phi_target = np.zeros((5, 3))
for i, floor in enumerate(floor_cables):
    rows = [sensor_index[c] for c in floor_cables[floor]]
    Phi_target[i, :] = Phi_full[rows, :].mean(axis=0)

# ---------------------------
# 4. Shear‐frame matrix assembly for 5-DOF
# ---------------------------
def shear_matrices(m, k):
    n = len(m)
    M = np.diag(m)
    K = np.zeros((n, n))
    for j in range(n):
        K[j, j] = k[j] + (k[j+1] if j+1<n else 0)
        if j+1 < n:
            K[j, j+1] = -k[j+1]
            K[j+1, j] = -k[j+1]
    return M, K

# ---------------------------
# 5. Residuals for 5‐DOF calibration
# ---------------------------
def residuals(x):
    k = x[:5]; m = x[5:]
    M, K = shear_matrices(m, k)
    w2, Phi_model = eigh(K, M)
    f_model = np.sqrt(w2)/(2*np.pi)
    freq_err = f_model[:3] - f_target     # first 3 modes
    mac_err = []
    for j in range(3):
        pm = Phi_model[:, j]
        pt = Phi_target[:, j]
        mac = (pm.dot(pt))**2/(pm.dot(pm)*pt.dot(pt))
        mac_err.append(1 - mac)
    return np.hstack((freq_err, mac_err))

# ---------------------------
# 6. Optimization (5 stiffness + 5 masses)
# ---------------------------
x0 = np.hstack((1e7*np.ones(5), 2e5*np.ones(5)))
lb = np.hstack((1e5*np.ones(5), 1e4*np.ones(5)))
ub = np.hstack((1e9*np.ones(5), 1e7*np.ones(5)))

res = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-8)

k_opt = res.x[:5]
m_opt = res.x[5:]
M_opt, K_opt = shear_matrices(m_opt, k_opt)
w2_opt, _ = eigh(K_opt, M_opt)
f_opt = np.sqrt(w2_opt)/(2*np.pi)

# ---------------------------
# 7. Results
# ---------------------------
print("Target freqs (Hz):   ", f_target)
print("Fitted freqs (Hz):   ", f_opt[:3])
print("Optimized k (N/m):   ", k_opt)
print("Optimized m (kg):    ", m_opt)
