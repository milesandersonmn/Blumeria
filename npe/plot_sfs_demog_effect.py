"""
Simulate SFS with alpha=1.641 under flat demography and compare to the
best-match variable-demography simulation.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from SBI import sim_summary_stats, stdout_to_log

ALPHA    = 1.6409
K        = 1.0      # flat k
N_SIMS   = 50
N_WORKERS = 7

# --- Best-match variable-demography simulation ---
x     = np.load('x.npy')
theta = np.load('theta.npy')
obs   = pd.read_csv('observed_sum_stats_SBI.csv').values.squeeze()

sfs_obs  = obs[:42]
dists    = np.sqrt(((x[:, :42] - sfs_obs) ** 2).sum(axis=1))
best     = np.argmin(dists)
sfs_best = x[best, :42]
t        = theta[best]
alpha_best, k_best = t[0], t[1]
Ne_mults = t[2:]

breakpoints  = [10, 50, 100, 200, 300, 400, 500, 1000, 10000, 100000, 1000000, 10000000]
window_lefts  = breakpoints[:11]
window_rights = breakpoints[1:12]

t_steps  = [0, 10]
ne_steps = [1.0, 1.0]
for left, right, ne in zip(window_lefts, window_rights, Ne_mults):
    t_steps  += [left, right]
    ne_steps += [ne, ne]


# --- Simulate flat demography ---
print(f"Running {N_SIMS} flat-demography simulations (alpha={ALPHA})...")
results = []
for i in range(N_SIMS):
    print(f"  sim {i+1}/{N_SIMS}", end='\r')
    with stdout_to_log('sim.log'):
        stats = sim_summary_stats(ALPHA, K,
                                  1, 1, 1, 1, 1, 1, 1,   # Ne1-Ne7
                                  1, 1, 1, 1)             # Ne8-Ne11
    results.append(np.array(stats[:42]))
print()

sfs_flat     = np.array(results)
sfs_flat_mean = sfs_flat.mean(axis=0)
sfs_flat_std  = sfs_flat.std(axis=0)
sfs_diff      = sfs_best - sfs_flat_mean

print(f"Flat SFS mean bin 1: {sfs_flat_mean[0]:.4f}  (variable: {sfs_best[0]:.4f})")

# --- Plot ---
bins = np.arange(1, 43)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: SFS comparison
ax1 = axes[0]
ax1.plot(bins, sfs_best,      's--', color='#0072B2', linewidth=1.8, markersize=4, label='Best-match (variable $N_e$)')
ax1.plot(bins, sfs_flat_mean, '^:',  color='#CC79A7', linewidth=1.8, markersize=4,
         label=f'Constant $N_e$ ($\\alpha={ALPHA:.3f}$, $n={N_SIMS}$)')
ax1.fill_between(bins, sfs_flat_mean - sfs_flat_std, sfs_flat_mean + sfs_flat_std,
                 color='#CC79A7', alpha=0.15)
ax1.set_xlabel('Derived allele count', fontsize=11)
ax1.set_ylabel('Proportion of segregating sites', fontsize=11)
ax1.set_title('Site frequency spectrum', fontsize=12)
ax1.legend(fontsize=8)

# Panel 2: SFS difference
ax2 = axes[1]
ax2.bar(bins, sfs_diff,
        color=['#0072B2' if v >= 0 else '#D55E00' for v in sfs_diff],
        alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel('Derived allele count', fontsize=11)
ax2.set_ylabel('Variable $N_e$ $-$ Constant $N_e$', fontsize=11)
ax2.set_title('SFS difference: demographic effect', fontsize=12)

plt.tight_layout()
plt.savefig('sfs_demog_effect.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved sfs_demog_effect.png')
