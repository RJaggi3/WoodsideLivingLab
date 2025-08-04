import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample_poly, coherence

# ----------------------------
# 1. Generate a synthetic signal
# ----------------------------
fs = 1000  # original sample rate (Hz)
t = np.arange(0, 1, 1/fs)  # 1 second of data
f_signal = 50  # frequency of sine wave (Hz)
# Create a sine wave with a little noise
signal = np.sin(2*np.pi * f_signal * t) + 0.05*np.random.randn(len(t))

# ----------------------------
# 2. Apply decimation/resampling methods
# ----------------------------
# Method 1: Using scipy.decimate (q=2, zero-phase filtering)
signal_decimate = decimate(signal, 2, ftype='iir', zero_phase=True)

# Method 2: Using resample_poly (up=1, down=2)
signal_poly = resample_poly(signal, up=1, down=2)

# Note: Both methods output signals at half the original sample rate.
# For a fair time-domain comparison, we upsample back to the original time grid using interpolation.
t_decimated = t[::2]  # time axis for decimated signals

signal_decimate_interp = np.interp(t, t_decimated, signal_decimate)
signal_poly_interp     = np.interp(t, t_decimated, signal_poly)

# ----------------------------
# 3. Define functions for error metrics
# ----------------------------

def compute_metrics(original, reconstructed):
    # Mean Squared Error (MSE)
    mse = np.mean((original - reconstructed)**2)
    # Normalized MSE (nMSE)
    nmse = mse / np.mean(original**2)
    
    error_signal = original - reconstructed
    # Signal-to-Noise Ratio (SNR) in dB
    snr = 10 * np.log10(np.mean(original**2) / np.mean(error_signal**2))
    
    # Log-Spectral Distance (LSD)
    # We'll compute FFTs and measure the root mean square difference in the log domain.
    fft_orig  = np.fft.rfft(original)
    fft_recon = np.fft.rfft(reconstructed)
    mag_orig  = np.abs(fft_orig)+1e-12  # avoid log(0)
    mag_recon = np.abs(fft_recon)+1e-12
    
    log_orig  = 20 * np.log10(mag_orig)
    log_recon = 20 * np.log10(mag_recon)
    lsd = np.sqrt(np.mean((log_orig - log_recon)**2))
    
    # Pearson Correlation Coefficient
    correlation = np.corrcoef(original, reconstructed)[0, 1]
    
    return mse, nmse, snr, lsd, correlation

metrics_decimate = compute_metrics(signal, signal_decimate_interp)
metrics_poly     = compute_metrics(signal, signal_poly_interp)

mse_dec, nmse_dec, snr_dec, lsd_dec, corr_dec = metrics_decimate
mse_poly, nmse_poly, snr_poly, lsd_poly, corr_poly = metrics_poly

# ----------------------------
# 4. Plot the time and frequency comparisons, and coherence
# ----------------------------

plt.figure(figsize=(12, 10))

# (a) Time-domain signals comparison
plt.subplot(3, 1, 1)
plt.plot(t, signal, label='Original Signal', color='black')
plt.plot(t, signal_decimate_interp, label='decimate (interp)', color='red', linestyle='--')
plt.plot(t, signal_poly_interp, label='resample_poly (interp)', color='blue', linestyle=':')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain: Original vs. Resampling Methods')
plt.legend()
plt.grid(True)

# (b) Frequency-domain amplitude spectra
plt.subplot(3, 1, 2)
f_orig   = np.fft.rfftfreq(len(signal), 1/fs)
mag_orig = np.abs(np.fft.rfft(signal))
mag_dec  = np.abs(np.fft.rfft(signal_decimate_interp))
mag_poly = np.abs(np.fft.rfft(signal_poly_interp))

plt.plot(f_orig, mag_orig, label='Original', color='black')
plt.plot(f_orig, mag_dec, label='decimate (interp)', color='red', linestyle='--')
plt.plot(f_orig, mag_poly, label='resample_poly (interp)', color='blue', linestyle=':')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain Comparison')
plt.legend()
plt.grid(True)

# (c) Coherence between original and each reconstructed signal
plt.subplot(3, 1, 3)
f_coh, Cxy_dec = coherence(signal, signal_decimate_interp, fs=fs)
_, Cxy_poly     = coherence(signal, signal_poly_interp, fs=fs)
plt.plot(f_coh, Cxy_dec, label='Coherence (decimate)', color='red', linestyle='--')
plt.plot(f_coh, Cxy_poly, label='Coherence (resample_poly)', color='blue', linestyle=':')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.title('Signal Coherence with Original')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# 5. Bar plot for error metrics
# ----------------------------
labels = ['MSE', 'nMSE', 'SNR (dB)', 'LSD (dB)', 'Correlation']
metrics_dec = [mse_dec, nmse_dec, snr_dec, lsd_dec, corr_dec]
metrics_poly= [mse_poly, nmse_poly, snr_poly, lsd_poly, corr_poly]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, metrics_dec, width, label='decimate')
rects2 = ax.bar(x + width/2, metrics_poly, width, label='resample_poly')

ax.set_title('Comparison of Error Metrics')
ax.set_ylabel('Value')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y')
plt.show()