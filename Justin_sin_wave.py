import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

# Create a sample signal (matching SciPy example)
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # 1-second time array
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 110 * t)  # 50Hz + 110Hz components

# Decimate by factor of 2 (using default IIR filter)
q = 2
x_decimated = signal.decimate(x, q, ftype='iir')

# Compute FFT helper function
def compute_fft(y, fs):
    n = len(y)
    yf = fft(y)
    xf = fftfreq(n, 1/fs)[:n//2]  # Only positive frequencies
    return xf, 2/n * np.abs(yf[0:n//2])

xf_orig, yf_orig = compute_fft(x, fs)
xf_dec, yf_dec = compute_fft(x_decimated, fs / q)

# Plot original time and frequency domain comparisons
plt.figure(figsize=(12, 6))

# Time domain comparison
plt.subplot(2, 1, 1)
plt.plot(t, x, 'b-', label='Original (fs=1000Hz)')
dec_t = t[::q]  # Decimated time points
plt.plot(dec_t, x_decimated, 'ro-', label=f'Decimated (fs={fs/q}Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Time Domain Comparison')

# Frequency domain comparison
plt.subplot(2, 1, 2)
plt.plot(xf_orig, yf_orig, 'b-', label='Original FFT')
plt.plot(xf_dec, yf_dec, 'r-', label='Decimated FFT')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Frequency Domain Comparison')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 1. Reconstruction via Zero Padding in the Frequency Domain
# ------------------------------------------------------------------------------
M = len(x_decimated)  # Length of decimated signal (e.g., 500)
N = len(x)            # Original length (e.g., 1000)

# Compute FFT of decimated signal
X_dec = fft(x_decimated)

# For an even-length signal, decompose the FFT into two parts:
if M % 2 == 0:
    # k is the index for the Nyquist component
    k = M // 2  
    # "Low frequencies" (including DC and Nyquist)
    low = X_dec[:k+1]  
    # "High frequencies" (those with negative frequency indices)
    high = X_dec[k+1:]
    # Calculate the number of zeros to insert so that the padded FFT has length N
    zeros_padding = np.zeros(N - M, dtype=complex)
    # Concatenate low frequencies, inserted zeros, then high frequencies
    X_padded = np.concatenate([low, zeros_padding, high])
else:
    # For odd-length signals (not the case here)
    k = (M + 1) // 2
    low = X_dec[:k]
    high = X_dec[k:]
    zeros_padding = np.zeros(N - M, dtype=complex)
    X_padded = np.concatenate([low, zeros_padding, high])

# Reconstruct the decimated signal (upsampled) from the padded FFT vector
x_reconstructed = np.real(ifft(X_padded))

# ------------------------------------------------------------------------------
# 2. Plot the reconstructed signal and its difference from the original
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 8))

# Subplot 1: Reconstructed signal in time domain
plt.subplot(3, 1, 1)
plt.plot(t, x_reconstructed, 'g.-', label='Reconstructed Signal (Zero-padded FFT)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Reconstructed Signal from Decimated FFT (Zero Padding)')

# 3. Compute the difference signal between the original and reconstructed signals
diff_signal = x - x_reconstructed
plt.subplot(3, 1, 2)
plt.plot(t, diff_signal, 'm.-', label='Difference: Original - Reconstructed')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude Difference')
plt.legend()
plt.title('Difference between Original and Reconstructed Signal (Time Domain)')

# 4. FFT of the difference signal and its plot
xf_diff, yf_diff = compute_fft(diff_signal, fs)
plt.subplot(3, 1, 3)
plt.plot(xf_diff, yf_diff, 'c-', label='FFT of Difference Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.title('FFT of Difference Signal')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 5. Calculate the Spectral Flatness Measure (SFM) of the difference signal
# ------------------------------------------------------------------------------
# The SFM is defined as:
#     SFM = (Geometric Mean of the magnitude spectrum) / (Arithmetic Mean of the magnitude spectrum).
# It is often given in dB as: 10 * log10(SFM).

# Compute FFT of the difference signal
diff_fft = fft(diff_signal)
# Use only the positive frequencies for SFM calculation
mag_diff = np.abs(diff_fft[:len(diff_fft)//2])
epsilon = 1e-12  # small constant to avoid log(0)

# Geometric mean calculation (log-transform to avoid numerical issues)
geom_mean = np.exp(np.mean(np.log(mag_diff + epsilon)))
# Arithmetic mean calculation
arith_mean = np.mean(mag_diff)
# Spectral flatness (linear and in dB)
sfm = geom_mean / arith_mean
sfm_db = 10 * np.log10(sfm)

print("Spectral Flatness Measure (linear):", sfm)
print("Spectral Flatness Measure (dB):", sfm_db)