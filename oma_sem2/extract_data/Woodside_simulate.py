import numpy as np
from scipy import linalg, signal

def Woodside_sim_gen() -> tuple:
    """
    Simulate a 4-DOF shear-type building with prescribed natural frequencies.
    Returns:
      Y_noisy : (N×4) array of acceleration outputs with measurement noise
      U       : (N×4) array of white-noise inputs
      (fn, xi, phi) : tuple
        fn  : array of natural frequencies [Hz], length=4
        xi  : damping ratio (scalar)
        phi : mass-normalized mode-shape matrix (4×4)
    """
    # 1) User-specified targets
    fn_target  = np.array([1.60, 1.85, 3.00, 21.60])  # Hz
    lam_target = (2 * np.pi * fn_target)**2          # rad²/s²

    # 2) Simulation parameters
    dof   = 4
    m_val = 25.91       # kg
    xi    = 0.02        # modal damping ratio
    fs    = 173.61      # sampling frequency, Hz
    dt    = 1.0 / fs    # time step, s
    N     = 51619       # total samples
    noise = 0.1         # relative noise level

    # 3) Mass matrix
    M = np.eye(dof) * m_val

    # 4) Get a consistent φ from a uniform-shear K_old
    K_old = np.zeros((dof, dof))
    k_val = 10000.0
    for i in range(dof):
        if i < dof - 1:
            K_old[i, i]     += k_val
            K_old[i, i+1]   += -k_val
            K_old[i+1, i]   += -k_val
            K_old[i+1, i+1] += k_val
        else:
            K_old[i, i] += k_val

    _, phi = linalg.eigh(K_old, M)
    # phi from eigh(K_old, M) is already mass‐normalized: phi.T @ M @ phi = I

    # 5) Rebuild K so that (K, M) ⇒ lam_target
    #    FIX: include trailing @ M to satisfy K·φ = M·φ·Λ_target
    K = M @ phi @ np.diag(lam_target) @ phi.T @ M

    # 6) Physical damping matrix C
    w_target  = np.sqrt(lam_target)                   # rad/s
    zeta_diag = 2 * w_target * xi
    C_modal   = np.diag(zeta_diag)
    C         = M @ phi @ C_modal @ phi.T @ M

    # 7) Continuous-time state-space A, B, C_full, D_full
    zero   = np.zeros((dof, dof))
    Idn    = np.eye(dof)
    A_cont = np.block([
        [ zero,                Idn],
        [-linalg.solve(M, K), -linalg.solve(M, C)]
    ])

    C_full = np.hstack([
        -linalg.solve(M, K),
        -linalg.solve(M, C)
    ])
    D_full = linalg.solve(M, np.eye(dof))

    B = np.vstack([zero, linalg.solve(M, np.eye(dof))])
    D = D_full

    # 8) Discretize
    sysd = signal.StateSpace(A_cont, B, C_full, D).to_discrete(dt)

    # 9) Simulate white-noise excitation
    t       = np.arange(N) * dt
    rng     = np.random.RandomState(12345)
    U_clean = rng.randn(N, dof)
    U_noise = noise * rng.randn(N, dof)
    U       = U_clean + U_noise

    _, Y, _ = signal.dlsim(sysd, U, t)

    # 10) Add measurement noise
    meas_noise_std = noise * np.std(Y, axis=0)
    Y_noisy        = Y + rng.randn(*Y.shape) * meas_noise_std

    return Y_noisy, U, (fn_target, xi, phi)


if __name__ == "__main__":
    Y_noisy, U, (fn, xi, phi) = Woodside_sim_gen()
    total_time = Y_noisy.shape[0] * (1.0 / 173.61)
    print(f"Prescribed natural frequencies [Hz]: {fn}")
    print(f"Damping ratio ξ: {xi}")
    print(f"Mode-shape matrix φ:\n{phi}")
    print(f"Simulated record length: {total_time:.2f} s ({Y_noisy.shape[0]} samples at 173.61 Hz)")