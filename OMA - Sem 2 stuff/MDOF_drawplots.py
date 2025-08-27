#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot synthetic accelerations + FFT into a multipage PDF
with 4 channels per page, each row has:
  [ time series | amplitude spectrum ],
and explicit numeric tick marks on every x‐axis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 1. Load the synthetic acceleration data
df = pd.read_csv("synthetic_accs.csv", index_col=0)
time = df.index.values
dt = time[1] - time[0]          # assume uniform spacing
N = len(time)

# 2. Prepare PDF settings
output_pdf = "acceleration_timeseries_with_fft.pdf"
channels = df.columns.tolist()
n_per_page = 4

with PdfPages(output_pdf) as pdf:
    # Loop through channels in blocks of 4
    for start in range(0, len(channels), n_per_page):
        block = channels[start : start + n_per_page]
        n_block = len(block)

        # Create a 4×2 grid of subplots, no sharex so each shows ticks
        fig, axes = plt.subplots(n_per_page, 2, figsize=(8.27, 11.69))

        # Hide any extra rows on the last page
        if n_block < n_per_page:
            for row in range(n_block, n_per_page):
                axes[row, 0].set_visible(False)
                axes[row, 1].set_visible(False)

        # Plot each channel
        for row, chan in enumerate(block):
            # time‐series plot (left column)
            ax_ts = axes[row, 0]
            ax_ts.plot(time, df[chan], color="C0", linewidth=0.7)
            ax_ts.set_title(f"{chan} — Time Series", loc="left", fontsize=10)
            ax_ts.set_ylabel("m/s²", fontsize=8)
            ax_ts.grid(True, linestyle="--", alpha=0.4)
            # explicit tick marks every 5 points
            ts_ticks = np.linspace(time[0], time[-1], num=6)
            ax_ts.set_xticks(ts_ticks)
            ax_ts.set_xticklabels([f"{t:.0f}" for t in ts_ticks], fontsize=7)
            ax_ts.set_xlabel("Time (s)", fontsize=8)

            # FFT plot (right column)
            ax_fft = axes[row, 1]
            A = np.fft.rfft(df[chan].values)
            freqs = np.fft.rfftfreq(N, dt)
            amp = 2.0 / N * np.abs(A)
            ax_fft.semilogy(freqs, amp, color="C2", linewidth=0.7)
            ax_fft.set_title(f"{chan} — Spectrum", loc="left", fontsize=10)
            ax_fft.set_ylabel("Amplitude", fontsize=8)
            ax_fft.grid(True, linestyle="--", alpha=0.4)
            # explicit FFT tick marks
            fft_ticks = np.linspace(freqs[0], freqs[-1], num=6)
            ax_fft.set_xticks(fft_ticks)
            ax_fft.set_xticklabels([f"{f:.1f}" for f in fft_ticks], fontsize=7)
            ax_fft.set_xlabel("Frequency (Hz)", fontsize=8)

        plt.tight_layout(pad=1.0)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Generated {output_pdf} with {len(channels)} channels ({n_per_page} per page).")