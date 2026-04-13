import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.load('x.npy')
theta = np.load('theta.npy')
obs = pd.read_csv('observed_sum_stats_SBI.csv').values.squeeze()

sfs_obs = obs[:42]
dists = np.sqrt(((x[:, :42] - sfs_obs) ** 2).sum(axis=1))
best = np.argmin(dists)
sfs_best = x[best, :42]
t = theta[best]
alpha, k = t[0], t[1]
Ne_mults = t[2:]  # Ne1-Ne11, relative to current Ne

print(f"Best match: simulation {best}, SFS distance = {dists[best]:.6f}")
print(f"alpha = {alpha:.4f}, k = {k:.4f}")
for i, ne in enumerate(Ne_mults):
    print(f"  Ne{i+1} = {ne:.4f}")

breakpoints = [10, 50, 100, 200, 300, 400, 500, 1000, 10000, 100000, 1000000, 10000000]
window_lefts  = breakpoints[:11]
window_rights = breakpoints[1:12]

# Include present-day window: 0-10 generations ago, Ne = 1.0 (reference)
t_steps = [0, 10]
ne_steps = [1.0, 1.0]
for left, right, ne in zip(window_lefts, window_rights, Ne_mults):
    t_steps += [left, right]
    ne_steps += [ne, ne]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- SFS ---
bins = np.arange(1, 43)
ax1.plot(bins, sfs_obs,  'o-', color='#D55E00', linewidth=1.8, markersize=4, label='Observed')
ax1.plot(bins, sfs_best, 's--', color='#0072B2', linewidth=1.8, markersize=4, label='Best-match simulation')
ax1.set_xlabel('Derived allele count', fontsize=11)
ax1.set_ylabel('Proportion of segregating sites', fontsize=11)
ax1.set_title('Site frequency spectrum', fontsize=12)
ax1.legend(fontsize=9)

# --- Ne trajectory ---
ax2.step(t_steps, ne_steps, where='post', color='#009E73', linewidth=2)
ax2.set_xscale('log')
ax2.set_xlim(max(window_rights), 1)
ax2.set_xlabel('Generations ago', fontsize=11)
ax2.set_ylabel(r'$N_e$ relative to present', fontsize=11)
ax2.set_title(
    f'Inferred $N_e$ trajectory ($\\alpha={alpha:.3f}$, $k={k:.2f}$)',
    fontsize=12
)

plt.tight_layout()
plt.savefig('best_match_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved best_match_plot.png')
