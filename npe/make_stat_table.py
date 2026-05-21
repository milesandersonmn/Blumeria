"""
Generate a publication-ready appendix table of all 195 summary statistics.
Outputs:
  summary_stats_table.csv   — for inspection / supplementary data file
  summary_stats_table.tex   — LaTeX longtable for appendix
"""

import csv

# ---------------------------------------------------------------------------
# Statistic definitions: (index, short_name, group, description)
# ---------------------------------------------------------------------------
rows = []

def add(idx, name, group, description):
    rows.append((idx + 1, name, group, description))  # 1-based for publication

# SFS bins 1-42
for i in range(42):
    add(i, f"SFS$_{{{i+1}}}$", "Site frequency spectrum",
        f"Proportion of segregating sites with {i+1} derived copies "
        f"(out of 43 haplotypes); denominator excludes monomorphic and fixed classes")

# AFS quantiles
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9"]):
    add(42 + qi, f"AFS$_{{q{q}}}$", "Site frequency spectrum",
        f"{q} quantile of the allele frequency spectrum across segregating sites")

# SFS symmetry ratio
add(47, "SFS symmetry", "Site frequency spectrum",
    "Ratio of high-frequency to low-frequency derived allele counts "
    r"(bins $> n/2$ divided by bins $\leq n/2$); 0 when low-frequency sum is zero")

# Tajima's D
add(48, r"$\bar{D}$",          "Tajima's D", "Mean Tajima's D across 30 windows")
add(49, r"$\mathrm{Var}(D)$",  "Tajima's D", "Variance of Tajima's D across windows")
add(50, r"$\sigma(D)$",        "Tajima's D", "Standard deviation of Tajima's D across windows")
add(51, r"$\mathrm{CV}(D)$",   "Tajima's D", r"Coefficient of variation of Tajima's D ($\sigma/\bar{x}$)")

# Hamming distances
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9"]):
    add(52 + qi, f"Ham$_{{q{q}}}$", "Hamming distance",
        f"{q} quantile of pairwise Hamming distances between haplotypes")
add(57, r"$\bar{\mathrm{Ham}}$",          "Hamming distance", "Mean pairwise Hamming distance")
add(58, r"$\sigma(\mathrm{Ham})$",        "Hamming distance", "Standard deviation of pairwise Hamming distances")
add(59, r"$\mathrm{Var}(\mathrm{Ham})$",  "Hamming distance", "Variance of pairwise Hamming distances")

# r^2
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"]):
    add(60 + qi, f"$r^2_{{q{q}}}$", "$r^2$ (LD)",
        f"{q} quantile of pairwise $r^2$ across chromosomes (singletons masked)")
add(67, r"$\bar{r}^2$",                  "$r^2$ (LD)", r"Mean $r^2$")
add(68, r"$\mathrm{Var}(r^2)$",          "$r^2$ (LD)", r"Variance of $r^2$")
add(69, r"$\sigma(r^2)$",                "$r^2$ (LD)", r"Standard deviation of $r^2$")
add(70, r"$\mathrm{CV}(r^2)$",           "$r^2$ (LD)", r"Coefficient of variation of $r^2$")
add(71, r"$\bar{r}^2 - \tilde{r}^2$",   "$r^2$ (LD)", r"Mean minus median $r^2$")
add(72, r"$P(r^2 \geq 1)$",             "$r^2$ (LD)", r"Proportion of $r^2$ values $\geq 1$")

# ILD
_ild_note = "interchromosomal LD (pairs drawn from different chromosomes)"
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"]):
    add(73 + qi, f"ILD$_{{q{q}}}$", "ILD",
        f"{q} quantile of {_ild_note}")
add(80, r"$\overline{\mathrm{ILD}}$",              "ILD", f"Mean {_ild_note}")
add(81, r"$\mathrm{Var}(\mathrm{ILD})$",           "ILD", f"Variance of {_ild_note}")
add(82, r"$\sigma(\mathrm{ILD})$",                 "ILD", f"Standard deviation of {_ild_note}")
add(83, r"$\mathrm{CV}(\mathrm{ILD})$",            "ILD", f"Coefficient of variation of {_ild_note}")
add(84, r"$\overline{\mathrm{ILD}} - \widetilde{\mathrm{ILD}}$", "ILD", f"Mean minus median of {_ild_note}")
add(85, r"$P(\mathrm{ILD} \geq 1)$",               "ILD", r"Proportion of ILD values $\geq 1$")

# r^2_norm
_norm_note = r"$r^2_\mathrm{max}$ as defined in \cite{VANLIERE_LD_2008}"
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"]):
    add(86 + qi, f"$r^2_{{\\mathrm{{norm}},q{q}}}$", r"$r^2_\mathrm{norm}$ (normalised LD)",
        f"{q} quantile of {_norm_note}")
add(93, r"$\bar{r}^2_\mathrm{norm}$",              r"$r^2_\mathrm{norm}$ (normalised LD)", f"Mean {_norm_note}")
add(94, r"$\mathrm{Var}(r^2_\mathrm{norm})$",      r"$r^2_\mathrm{norm}$ (normalised LD)", f"Variance of {_norm_note}")
add(95, r"$\sigma(r^2_\mathrm{norm})$",            r"$r^2_\mathrm{norm}$ (normalised LD)", f"Standard deviation of {_norm_note}")
add(96, r"$\mathrm{CV}(r^2_\mathrm{norm})$",       r"$r^2_\mathrm{norm}$ (normalised LD)", f"Coefficient of variation of {_norm_note}")
add(97, r"$\bar{r}^2_\mathrm{norm} - \tilde{r}^2_\mathrm{norm}$", r"$r^2_\mathrm{norm}$ (normalised LD)", f"Mean minus median of {_norm_note}")
add(98, r"$P(r^2_\mathrm{norm} \geq 1)$",          r"$r^2_\mathrm{norm}$ (normalised LD)", r"Proportion of normalised $r^2 \geq 1$")

# ILD_norm
_ild_norm_note = r"interchromosomal LD using $r^2_\mathrm{max}$ \cite{VANLIERE_LD_2008}"
for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"]):
    add(99 + qi, f"ILD$_{{\\mathrm{{norm}},q{q}}}$", r"ILD$_\mathrm{norm}$ (normalised ILD)",
        f"{q} quantile of {_ild_norm_note}")
add(106, r"$\overline{\mathrm{ILD}}_\mathrm{norm}$",              r"ILD$_\mathrm{norm}$ (normalised ILD)", f"Mean {_ild_norm_note}")
add(107, r"$\mathrm{Var}(\mathrm{ILD}_\mathrm{norm})$",           r"ILD$_\mathrm{norm}$ (normalised ILD)", f"Variance of {_ild_norm_note}")
add(108, r"$\sigma(\mathrm{ILD}_\mathrm{norm})$",                 r"ILD$_\mathrm{norm}$ (normalised ILD)", f"Standard deviation of {_ild_norm_note}")
add(109, r"$\mathrm{CV}(\mathrm{ILD}_\mathrm{norm})$",            r"ILD$_\mathrm{norm}$ (normalised ILD)", f"Coefficient of variation of {_ild_norm_note}")
add(110, r"$\overline{\mathrm{ILD}}_\mathrm{norm} - \widetilde{\mathrm{ILD}}_\mathrm{norm}$", r"ILD$_\mathrm{norm}$ (normalised ILD)", f"Mean minus median of {_ild_norm_note}")
add(111, r"$P(\mathrm{ILD}_\mathrm{norm} \geq 1)$",               r"ILD$_\mathrm{norm}$ (normalised ILD)", r"Proportion of normalised ILD $\geq 1$")

# Normalised Tajima's D
add(112, r"$\bar{D}_\mathrm{norm}$",         "Normalised Tajima's D",
    r"Mean of $(\theta_\pi - \theta_W)/\theta_\pi$ across 30 windows")
add(113, r"$\sigma(D_\mathrm{norm})$",       "Normalised Tajima's D",
    r"Standard deviation of normalised Tajima's D across windows")
add(114, r"$\mathrm{CV}(D_\mathrm{norm})$",  "Normalised Tajima's D",
    r"Coefficient of variation of normalised Tajima's D")

# LD frequency spectra (r^2, ILD, r^2_norm, ILD_norm) — 10 bins + diff each
bins = [f"[{0.1*i:.1f}, {0.1*(i+1):.1f})" for i in range(10)]
for stat, base, grp, extra in [
    ("$r^2$",               115, "LD frequency spectrum ($r^2$)",              ""),
    ("ILD",                 126, "LD frequency spectrum (ILD)",                " (interchromosomal LD)"),
    ("$r^2_\\mathrm{norm}$", 137, r"LD frequency spectrum ($r^2_\mathrm{norm}$)",
     r"; $r^2_\mathrm{max}$ as defined in \cite{VANLIERE_LD_2008}"),
    ("ILD$_\\mathrm{norm}$", 148, r"LD frequency spectrum (ILD$_\mathrm{norm}$)",
     r"; interchromosomal LD using $r^2_\mathrm{max}$ \cite{VANLIERE_LD_2008}"),
]:
    for bi, b in enumerate(bins):
        add(base + bi, f"{stat} LD-spec {b}", grp,
            f"Proportion of {stat} values in bin {b}{extra}")
    add(base + 10, f"{stat} LD-spec diff", grp,
        f"Difference between proportion in [0.0, 0.1) and [0.9, 1.0] for {stat}{extra} "
        "(unlinked minus fully linked)")

# Adjacent-site weighted/unweighted r^2 and r^2_norm
_wt_note = "weighted by inverse physical distance (bp) between adjacent sites"
for stat, base, grp, norm_note in [
    ("adj-$r^2$",               159, "Adjacent-site $r^2$",              ""),
    ("adj-$r^2_\\mathrm{norm}$", 177, r"Adjacent-site $r^2_\mathrm{norm}$",
     r"; $r^2_\mathrm{max}$ as defined in \cite{VANLIERE_LD_2008}"),
]:
    for mode, mbase, mlabel in [("weighted", base, "wtd"), ("unweighted", base + 9, "unwtd")]:
        wt_desc = f", {_wt_note}" if mode == "weighted" else ""
        for qi, q in enumerate(["0.1", "0.3", "0.5", "0.7", "0.9"]):
            add(mbase + qi, f"{stat} {mlabel} $q_{{{q}}}$", grp,
                f"{q} quantile of adjacent-site {stat}{norm_note}{wt_desc}")
        add(mbase + 5, f"{stat} {mlabel} mean",     grp, f"Mean adjacent-site {stat}{norm_note}{wt_desc}")
        add(mbase + 6, f"{stat} {mlabel} std",      grp, f"Standard deviation of adjacent-site {stat}{norm_note}{wt_desc}")
        add(mbase + 7, f"{stat} {mlabel} CV",       grp, f"Coefficient of variation of adjacent-site {stat}{norm_note}{wt_desc}")
        add(mbase + 8, f"{stat} {mlabel} mean-med", grp, f"Mean minus median of adjacent-site {stat}{norm_note}{wt_desc}")

assert len(rows) == 195, f"Expected 195 rows, got {len(rows)}"

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
with open("summary_stats_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Statistic", "Group", "Description"])
    for r in rows:
        writer.writerow(r)
print("Saved summary_stats_table.csv")

# ---------------------------------------------------------------------------
# LaTeX longtable output
# ---------------------------------------------------------------------------
tex_lines = []
tex_lines.append(r"\small")
tex_lines.append(r"\begin{longtable}{@{}rlp{3cm}p{6cm}@{}}")
tex_lines.append(r"\caption{Summary statistics used in simulation-based inference. "
                 r"All 195 statistics are computed for each simulated and observed dataset. "
                 r"$n = 43$ haplotypes; singletons are masked for LD statistics. "
                 r"CV $= \sigma / \bar{x}$.} "
                 r"\label{tab:summary_stats} \\")
tex_lines.append(r"\toprule")
tex_lines.append(r"\# & Statistic & Group & Description \\")
tex_lines.append(r"\midrule")
tex_lines.append(r"\endfirsthead")
tex_lines.append(r"\multicolumn{4}{l}{\small\itshape (Table \ref{tab:summary_stats} continued)} \\")
tex_lines.append(r"\toprule")
tex_lines.append(r"\# & Statistic & Group & Description \\")
tex_lines.append(r"\midrule")
tex_lines.append(r"\endhead")
tex_lines.append(r"\midrule")
tex_lines.append(r"\multicolumn{4}{r}{\small\itshape Continued on next page} \\")
tex_lines.append(r"\endfoot")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\endlastfoot")

prev_group = None
for idx, name, group, description in rows:
    # Add a small visual separator between groups
    if group != prev_group and prev_group is not None:
        tex_lines.append(r"\midrule")
    prev_group = group
    # Escape % and & in description (not in math mode)
    desc = description.replace("%", r"\%")
    tex_lines.append(f"{idx} & {name} & {group} & {desc} \\\\")

tex_lines.append(r"\end{longtable}")

with open("summary_stats_table.tex", "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print("Saved summary_stats_table.tex")
print(f"Total statistics: {len(rows)}")
