import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Define SDOF Parameters
m = 1.0                     # mass [kg]
k = (2*np.pi*5)**2 * m      # stiffness for fn = 5 Hz
c = 2 * m * (2*np.pi*5) * 0.02  # damping for zeta = 0.02

# Derived quantities
fn = 1/(2*np.pi) * np.sqrt(k/m)
zeta = c/(2*np.sqrt(k*m))

# Time settings
fs = 200.0                  # sampling rate [Hz]
dt = 1/fs                   # time step [s]
T = 300.0                    # total duration [s]
t = np.arange(0, T, dt)     # time vector

# Define Excitation (white noise)
np.random.seed(42)
f = 0.1 * np.random.randn(len(t))  # force amplitude scaled

# Equation of motion: m*x'' + c*x' + k*x = f(t)
def sdof_ode(ti, yi):
    x, v = yi
    # interpolate force at time ti
    fi = np.interp(ti, t, f)
    dxdt = v
    dvdt = (fi - c*v - k*x) / m
    return [dxdt, dvdt]

# Integrate ODE
y0 = [0.0, 0.0]  # initial displacement and velocity
sol = solve_ivp(sdof_ode, [t[0], t[-1]], y0, t_eval=t, method='RK45')

x = sol.y[0]
v = sol.y[1]
a = np.gradient(v, dt)  # approximate acceleration

# 2.5 Add sensor effects (Gaussian noise + quantisation)
noise_level = 0.02
x_noisy = x + noise_level * np.std(x) * np.random.randn(len(x))

adc_bits = 16
adc_range = np.max(np.abs(x_noisy))
quant = 2*adc_range / (2**adc_bits)
x_noisy_q = (quant * np.round(x_noisy/quant))

# 2.6 Package into DataFrame
df = pd.DataFrame({
    'time': t,
    'disp_true': x,
    'vel_true': v,
    'acc_true': a,
    'disp_noisy': x_noisy_q
})

# 2.7 Save to disk
df.to_csv('sdof_dataset.csv', index=False)

with h5py.File('sdof_dataset.h5', 'w') as hf:
    grp = hf.create_group('sdof')
    for col in df.columns:
        grp.create_dataset(col, data=df[col].values)

# 2.8 Quick Plot
plt.figure(figsize=(10,4))
plt.plot(t, x, label='True displacement')
#plt.plot(t, x_noisy_q, '.', ms=1, alpha=0.5, label='Noisy + quantised')
plt.xlabel('Time [s]')
plt.ylabel('Displacement')
plt.legend()
plt.tight_layout()
plt.show()