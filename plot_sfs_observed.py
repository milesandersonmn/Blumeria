"""
Plot the observed SFS from the saved summary stats CSV.
Run with: python plot_sfs_observed.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sample_size = 43

df = pd.read_csv("observed_sum_stats_SBI.csv")
sfs = df.iloc[0, :sample_size - 1].values  # SFS bins 1-42

derived_counts = np.arange(1, sample_size)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(derived_counts, sfs, color='#0072B2', alpha=0.8)
ax.set_xlabel("Derived allele count")
ax.set_ylabel("Proportion of segregating sites")
ax.set_title("Site Frequency Spectrum")
plt.tight_layout()
plt.savefig("sfs_observed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sfs_observed.png")
