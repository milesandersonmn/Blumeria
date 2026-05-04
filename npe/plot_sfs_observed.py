"""
Plot the observed SFS from the saved summary stats CSV.
Run with: python npe/plot_sfs_observed.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
RESULTS_DIR  = os.path.join(_BASE_DIR, "results")
FIGURES_DIR  = os.path.join(_BASE_DIR, "figures")

sample_size = 43

df = pd.read_csv(os.path.join(RESULTS_DIR, "observed_sum_stats_SBI.csv"))
sfs = df.iloc[0, :sample_size - 1].values  # SFS bins 1-42

derived_counts = np.arange(1, sample_size)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(derived_counts, sfs, color='#0072B2', alpha=0.8)
ax.set_xlabel("Derived allele count")
ax.set_ylabel("Proportion of segregating sites")
ax.set_title("Site Frequency Spectrum")
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "sfs_observed.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
