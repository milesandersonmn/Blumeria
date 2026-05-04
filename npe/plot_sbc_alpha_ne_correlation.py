"""
Test whether the NPE systematically confounds alpha with Ne demographic history,
using Simulation-Based Calibration (SBC) output.

For each SBC trial the model was given a simulation generated from a known
(true) parameter set drawn from the prior. We then ask: does the posterior mean
alpha inferred by the model correlate with the true Ne values used to generate
that simulation?

This is the correct test for confounding: the true Ne values span the full prior
range (0.1-5) across 500 independent trials, so we are testing whether the
inference method itself shifts its alpha estimate based on the true demographic
history -- not whether alpha and Ne happen to co-vary within a single posterior.

Input files (produced by: python SBI.py --sbc-only --posterior-file posterior_nsf.pt):
    sbc_true_thetas.csv  -- true parameter values drawn from prior (500 x 13)
    sbc_post_means.csv   -- posterior mean for each parameter per trial (500 x 13)

Usage:
    python plot_sbc_alpha_ne_correlation.py
    python plot_sbc_alpha_ne_correlation.py --out sbc_alpha_ne_corr.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

WINDOW_LABELS = {
    'Ne1':  '10–50 gen',
    'Ne2':  '50–100 gen',
    'Ne3':  '100–200 gen',
    'Ne4':  '200–300 gen',
    'Ne5':  '300–400 gen',
    'Ne6':  '400–500 gen',
    'Ne7':  '500–1000 gen',
    'Ne8':  '1–10 kgen',
    'Ne9':  '10–100 kgen',
    'Ne10': '100 k–1 Mgen',
    'Ne11': '>1 Mgen',
}

N_PERMUTATIONS = 1000


def permutation_r(x, y, n=N_PERMUTATIONS, seed=42):
    rng = np.random.default_rng(seed)
    obs_r, _ = pearsonr(x, y)
    perm_r = np.array([pearsonr(rng.permutation(x), y)[0] for _ in range(n)])
    p = np.mean(np.abs(perm_r) >= np.abs(obs_r))
    ci_lo, ci_hi = np.percentile(perm_r, [2.5, 97.5])
    return obs_r, p, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--true-thetas", default="sbc_true_thetas.csv",
                        help="CSV of true SBC parameter values (default: sbc_true_thetas.csv)")
    parser.add_argument("--post-means", default="sbc_post_means.csv",
                        help="CSV of posterior means per SBC trial (default: sbc_post_means.csv)")
    parser.add_argument("--out", default="sbc_alpha_ne_correlation.png",
                        help="Output PNG path (default: sbc_alpha_ne_correlation.png)")
    args = parser.parse_args()

    true_df = pd.read_csv(args.true_thetas)
    post_df = pd.read_csv(args.post_means)
    ne_cols = [c for c in true_df.columns if c.startswith('Ne')]

    inferred_alpha = post_df['alpha'].values

    # --- Per-window correlation: true Ne vs inferred alpha ---
    results = []
    print(f"\n{'Window':<8} {'Time range':<16} {'Pearson r':>10} {'p-value':>10}")
    print('-' * 48)
    null_ci_lo_all, null_ci_hi_all = [], []
    for c in ne_cols:
        true_ne = true_df[c].values
        r, p, ci_lo, ci_hi = permutation_r(true_ne, inferred_alpha)
        null_ci_lo_all.append(ci_lo)
        null_ci_hi_all.append(ci_hi)
        label = WINDOW_LABELS.get(c, c)
        results.append(dict(window=c, label=label, r=r, p=p,
                            perm_ci_lo=ci_lo, perm_ci_hi=ci_hi))
        sig = '*' if p < 0.05 else ' '
        print(f"{c:<8} {label:<16} {r:>10.4f} {p:>9.3f}{sig}")

    res = pd.DataFrame(results)
    null_ci_lo = np.mean(null_ci_lo_all)
    null_ci_hi = np.mean(null_ci_hi_all)

    # Also: true alpha vs inferred alpha (sanity check)
    r_alpha, p_alpha, _, _ = permutation_r(true_df['alpha'].values, inferred_alpha)
    print(f"\nNPE performance — true alpha vs inferred alpha: r={r_alpha:.4f}, p={p_alpha:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Pearson r per window
    ax = axes[0]
    x_pos = np.arange(len(ne_cols))
    ax.axhspan(null_ci_lo, null_ci_hi, alpha=0.15, color='gray',
               label='Permutation null 95% CI')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    colors = ['#D55E00' if p < 0.05 else '#0072B2' for p in res['p']]
    ax.bar(x_pos, res['r'], color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{r['window']}\n({r['label']})" for _, r in res.iterrows()],
        fontsize=8, rotation=35, ha='right', rotation_mode='anchor'
    )
    ax.set_ylabel("Pearson $r$ (true $N_e$ vs inferred $\\alpha$)", fontsize=10)
    ax.set_xlabel("$N_e$ time window", fontsize=10)
    ax.set_title(
        "Confounding of $\\alpha$ inference by true demographic history\n"
        "across 500 independent SBC trials (prior-spanning)",
        fontsize=10
    )
    ax.set_ylim(-0.3, 0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#D55E00', alpha=0.85, label='$p$ < 0.05 (permutation)'),
        Patch(facecolor='#0072B2', alpha=0.85, label='$p$ ≥ 0.05'),
        Patch(facecolor='gray', alpha=0.15, label='Permutation null 95% CI'),
    ], fontsize=8)

    # Right: true alpha vs inferred alpha (NPE performance validation)
    ax = axes[1]
    ax.scatter(true_df['alpha'].values, inferred_alpha,
               alpha=0.3, s=15, color='#0072B2')
    lims = [min(true_df['alpha'].min(), inferred_alpha.min()) - 0.05,
            max(true_df['alpha'].max(), inferred_alpha.max()) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line')
    ax.set_xlabel(r'True $\alpha$ (drawn from prior)', fontsize=10)
    ax.set_ylabel(r'Inferred $\alpha$ (NPE posterior mean)', fontsize=10)
    ax.set_title(
        f'NPE performance: true vs inferred $\\alpha$\n'
        f'across 500 prior-spanning SBC trials ($r$ = {r_alpha:.3f}, $p$ < 0.001)',
        fontsize=10
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
