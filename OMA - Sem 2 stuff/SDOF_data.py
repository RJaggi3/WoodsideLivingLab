import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# 1. Define SDOF parameters
m = 1.0                           # mass [kg]
k = (2 * np.pi * 5)**2 * m       # stiffness for fn = 5 Hz
c = 2 * m * (2 * np.pi * 5) * 0.02  # damping for zeta = 0.02

# natural frequency (Hz) for reference
fn = 1/(2 * np.pi) * np.sqrt(k/m)

# 2. Time vector
fs = 200.0            # sampling rate [Hz]
dt = 1/fs             # time step [s]
T = 300.0             # total duration [s]
t = np.arange(0, T, dt)
N = len(t)

# 3. White-noise excitation
np.random.seed(42)
f = 0.1 * np.random.randn(N)

# 4. ODE definition: m*x'' + c*x' + k*x = f(t)
def sdof_ode(ti, yi):
    x, v = yi
    fi = np.interp(ti, t, f)
    dxdt = v
    dvdt = (fi - c * v - k * x) / m
    return [dxdt, dvdt]

# 5. Integrate
y0 = [0.0, 0.0]
sol = solve_ivp(sdof_ode, (t[0], t[-1]), y0, t_eval=t, method='RK45')
x = sol.y[0]
v = sol.y[1]

# 6. Compute acceleration from the EOM residual
a = (f - c * v - k * x) / m

# 7. Save acceleration time series
df = pd.DataFrame({'time': t, 'acc_true': a})
df.to_csv('sdof_acceleration.csv', index=False)
with h5py.File('sdof_acceleration.h5', 'w') as hf:
    grp = hf.create_group('sdof')
    grp.create_dataset('time', data=t)
    grp.create_dataset('acc_true', data=a)

# 8. FFT of acceleration
#    single-sided spectrum
A = np.fft.rfft(a)
freqs = np.fft.rfftfreq(N, dt)
amp = 2.0 / N * np.abs(A)

# find dominant peak
peak_idx = np.argmax(amp)
peak_freq = freqs[peak_idx]
print(f"Designed fn = {fn:.3f} Hz; FFT peak = {peak_freq:.3f} Hz")

# 9. Plotting
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, a, color='C1')
plt.title('Time Series: Acceleration')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')

plt.subplot(1,2,2)
plt.semilogy(freqs, amp, color='C2')

ax2 = plt.subplot(1,2,2)
ax2.semilogy(freqs, amp, color='C2')
ax2.set_xscale('log')          
ax2.axvline(fn, color='k', linestyle='--', label=f'fn = {fn:.1f} Hz')

ax2.set_title('FFT of Acceleration')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Amplitude')
ax2.legend()



plt.tight_layout()
plt.show()
