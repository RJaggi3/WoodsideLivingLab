import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CSV
df = pd.read_csv("ssi_results.csv")

# 2. Extract reference values (decimate, q=1) for each mode
ref = (
    df
    .query("Method == 'decimate' and q == 1")
    .set_index("Mode")[["Frequency", "Damping"]]
    .rename(columns={"Frequency": "Freq_ref", "Damping": "Zeta_ref"})
)

# 3. Merge reference back into full DataFrame
df = df.join(ref, on="Mode")

# 4. Compute percent errors
df["FreqErr%"]  = (df["Frequency"] - df["Freq_ref"]).abs() / df["Freq_ref"] * 100
df["ZetaErr%"]  = (df["Damping"]   - df["Zeta_ref"]).abs()   / df["Zeta_ref"] * 100

# 5. Aggregate across modes: mean error & mean MAC for each Method×q
agg = (
    df
    .groupby(["Method", "q"])
    .agg(
      FreqErr_mean = ("FreqErr%", "mean"),
      ZetaErr_mean = ("ZetaErr%", "mean"),
      MAC_mean     = ("MAC",    "mean")
    )
    .reset_index()
)

# 6. Plotting
sns.set(context="talk", style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# a) Frequency error vs q
sns.lineplot(
    data=agg,
    x="q", y="FreqErr_mean", hue="Method",
    marker="o", ax=axes[0]
)
axes[0].set_title("Avg. Frequency Error (%)")
axes[0].set_xlabel("Decimation Factor q")
axes[0].set_ylabel("Freq Error (%)")
axes[0].legend(title="Method")

# b) Damping error vs q
sns.lineplot(
    data=agg,
    x="q", y="ZetaErr_mean", hue="Method",
    marker="o", ax=axes[1]
)
axes[1].set_title("Avg. Damping Ratio Error (%)")
axes[1].set_xlabel("Decimation Factor q")
axes[1].set_ylabel("ζ Error (%)")
axes[1].legend_.remove()  # shared legend

# c) Mean MAC vs q
sns.lineplot(
    data=agg,
    x="q", y="MAC_mean", hue="Method",
    marker="o", ax=axes[2]
)
axes[2].set_title("Avg. Modal Assurance Criterion")
axes[2].set_xlabel("Downsampling  q")
axes[2].set_ylabel("MAC")
axes[2].set_ylim(0.9, 1.01)
axes[2].legend_.remove()

plt.tight_layout()
plt.show()