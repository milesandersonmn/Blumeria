"""
Simulate a chromosome with sub-telomeric and centromeric regions using the
best-fit alpha and demography from the NPE inference, and produce output files
for selective sweep detection with SweeD and RaisD.

Chromosome structure (left to right):
  [0, SUBTEL_LEN)              sub-telomeric region  -- r_eff (normal)
  [SUBTEL_LEN, SUBTEL_LEN+CEN_LEN)  centromere        -- r_centromere (low)
  [SUBTEL_LEN+CEN_LEN, CHROM_LEN)   sub-telomeric region -- r_eff (normal)

Output files:
  sweep_sim.vcf      -- VCF for SweeD (requires VCF input) and RaisD
  sweep_sim.ms       -- ms-format for SweeD --ms flag

Usage:
    python sim_sweep_detection.py
    python sim_sweep_detection.py --chrom-len 5000000 --n-reps 1 --out-prefix sweep_sim
"""

import argparse
import math
import numpy as np
import pandas as pd
import msprime

# ---- Chromosome / rate parameters ----
DEFAULT_CHROM_LEN   = 5_000_000   # total chromosome length (bp)
DEFAULT_SUBTEL_LEN  = 2_000_000   # sub-telomeric region on each side (bp)
DEFAULT_CEN_LEN     = 1_000_000   # centromere length (bp)
                                   # SUBTEL + CEN + SUBTEL = CHROM_LEN

MU                  = 5e-7        # mutation rate per mitotic generation
R_PER_MEIOSIS       = 3.37e-7     # recombination rate per meiosis (sub-telomeric)
R_CENTROMERE_FACTOR = 0.01        # centromere r as fraction of sub-telomeric r
NE                  = 40_000      # base effective population size
N_SAMPLES           = 43          # haploid sample size
BREAKPOINTS         = [10, 50, 100, 200, 300, 400, 500,
                       1000, 10000, 100000, 1000000]


def get_best_match_params():
    """Load best-fit alpha and Ne multipliers from x.npy / theta.npy."""
    x     = np.load("x.npy")
    theta = np.load("theta.npy")
    obs   = pd.read_csv("observed_sum_stats_SBI.csv").values.squeeze()
    sfs_obs = obs[:42]
    dists   = np.sqrt(((x[:, :42] - sfs_obs) ** 2).sum(axis=1))
    best    = np.argmin(dists)
    t       = theta[best]
    alpha   = float(t[0])
    k       = float(t[1])
    ne_mults = t[2:].tolist()   # Ne1-Ne11
    print(f"Best-match simulation index: {best}  (SFS distance = {dists[best]:.6f})")
    print(f"  alpha = {alpha:.6f},  k = {k:.6f}")
    for i, ne in enumerate(ne_mults):
        print(f"  Ne{i+1} = {ne:.6f}")
    return alpha, k, ne_mults


def build_rate_map(chrom_len, subtel_len, cen_len, r_eff, r_centromere):
    """
    Build msprime RateMap for a single chromosome with:
      - sub-telomeric region (0 to subtel_len): r_eff
      - centromere (subtel_len to subtel_len+cen_len): r_centromere
      - sub-telomeric region (subtel_len+cen_len to chrom_len): r_eff
    """
    cen_start = subtel_len
    cen_end   = subtel_len + cen_len
    assert cen_end <= chrom_len, \
        f"Centromere end ({cen_end}) exceeds chromosome length ({chrom_len})"

    positions = [0, cen_start, cen_end, chrom_len]
    rates     = [r_eff, r_centromere, r_eff]
    return msprime.RateMap(position=positions, rate=rates)


def build_demography(ne_mults, ne_base=NE):
    """Build msprime Demography from Ne multipliers."""
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=ne_base)
    for t, ne_mult in zip(BREAKPOINTS, ne_mults):
        demography.add_population_parameters_change(
            time=t, population="A", initial_size=ne_base * ne_mult
        )
    return demography


def count_subtel_sites(mts, subtel_len, cen_len):
    """Count segregating sites in sub-telomeric regions only (excluding centromere)."""
    cen_start = subtel_len
    cen_end   = subtel_len + cen_len
    positions = mts.tables.sites.position
    in_subtel = (positions < cen_start) | (positions >= cen_end)
    return int(in_subtel.sum())


def simulate(alpha, ne_mults, rate_map, subtel_len, cen_len,
             target_snps_per_mb=1000, n_samples=N_SAMPLES,
             mu=MU, ne_base=NE, random_seed=None):
    """
    Run simulation with rejection sampling to target ~target_snps_per_mb SNPs
    per Mb in the sub-telomeric regions. The centromere SNP count is free to vary.
    Ne is rescaled each attempt until the sub-telomeric site count falls within
    the acceptance window.
    """
    chrom_len     = subtel_len * 2 + cen_len
    subtel_mb     = (chrom_len - cen_len) / 1e6          # total sub-telomeric Mb
    target_s      = target_snps_per_mb * subtel_mb        # expected total sub-tel sites
    target_min    = int(target_s * 0.99)
    target_max    = int(target_s * 1.01)

    demography = build_demography(ne_mults, ne_base)
    ne_current = ne_base
    attempt    = 0

    print(f"  Target sub-telomeric SNPs: {target_min}–{target_max} "
          f"({target_snps_per_mb}/Mb over {subtel_mb:.1f} Mb)")

    while True:
        attempt += 1
        # Rebuild demography with current Ne
        demography = build_demography(ne_mults, ne_current)
        ts = msprime.sim_ancestry(
            samples=n_samples,
            demography=demography,
            recombination_rate=rate_map,
            model=msprime.BetaCoalescent(alpha=alpha),
            ploidy=1,
            random_seed=random_seed,
        )
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=random_seed)
        S_subtel = count_subtel_sites(mts, subtel_len, cen_len)
        S_total  = mts.num_sites

        if S_subtel == 0:
            ne_current *= 2
            continue
        if S_subtel < target_min:
            ne_current = (target_min / S_subtel) * ne_current
            continue
        if S_subtel > target_max:
            ne_current = (target_max / S_subtel) * ne_current
            continue

        # Accepted
        S_cen = S_total - S_subtel
        print(f"  Accepted on attempt {attempt}: "
              f"S_subtel = {S_subtel}, S_centromere = {S_cen}, "
              f"S_total = {S_total}, Ne = {ne_current:.0f}")
        return mts


def write_vcf(mts, out_path, contig_name="chrom1"):
    """
    Write VCF file with ancestral allele (AA) in the INFO field.
    Compatible with SweeD --vcf and RaisD.

    In msprime simulations the REF allele is always the ancestral state, so
    AA=REF is correct for all sites. SweeD and RaisD require AA to polarise
    the SFS for sweep detection.
    """
    import io
    buf = io.StringIO()
    mts.write_vcf(buf, contig_id=contig_name)
    buf.seek(0)

    with open(out_path, "w") as out:
        for line in buf:
            if line.startswith("##FORMAT"):
                # Insert AA INFO header line before FORMAT lines
                out.write('##INFO=<ID=AA,Number=1,Type=String,'
                          'Description="Ancestral allele">\n')
                out.write(line)
            elif line.startswith("#") or line.strip() == "":
                out.write(line)
            else:
                fields = line.rstrip("\n").split("\t")
                ref = fields[3]
                # INFO field is index 7; tskit writes "." by default
                fields[7] = f"AA={ref}"
                out.write("\t".join(fields) + "\n")

    print(f"Saved {out_path}  ({mts.num_sites} sites, {mts.num_samples} samples, AA field added)")


def write_ms(mts, out_path, chrom_len):
    """
    Write ms-format file compatible with SweeD --ms.

    ms format:
      //
      segsites: S
      positions: p1 p2 ... pS   (in [0,1])
      haplotype matrix (n_samples rows, S columns)
    """
    tables  = mts.dump_tables()
    sites   = tables.sites
    S       = len(sites)
    pos_norm = sites.position / chrom_len   # normalise to [0,1]

    # Haplotype matrix: rows = samples, cols = sites
    G = mts.genotype_matrix()   # shape (S, n_samples), transpose for ms

    with open(out_path, "w") as f:
        f.write(f"ms {mts.num_samples} 1\n")
        f.write("//\n")
        f.write(f"segsites: {S}\n")
        f.write("positions: " + " ".join(f"{p:.6f}" for p in pos_norm) + "\n")
        for sample_idx in range(mts.num_samples):
            row = "".join(str(int(G[site_idx, sample_idx]))
                          for site_idx in range(S))
            f.write(row + "\n")
    print(f"Saved {out_path}  ({S} segregating sites)")


def write_region_bed(subtel_len, cen_len, chrom_len, out_path, contig_name="chrom1"):
    """
    Write a BED file annotating the three genomic regions.
    Useful for downstream filtering or visualisation.
    """
    cen_start = subtel_len
    cen_end   = subtel_len + cen_len
    with open(out_path, "w") as f:
        f.write(f"{contig_name}\t0\t{cen_start}\tsub-telomeric_left\n")
        f.write(f"{contig_name}\t{cen_start}\t{cen_end}\tcentromere\n")
        f.write(f"{contig_name}\t{cen_end}\t{chrom_len}\tsub-telomeric_right\n")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom-len", type=int, default=DEFAULT_CHROM_LEN,
                        help=f"Total chromosome length in bp (default: {DEFAULT_CHROM_LEN})")
    parser.add_argument("--subtel-len", type=int, default=DEFAULT_SUBTEL_LEN,
                        help=f"Sub-telomeric region length on each side in bp (default: {DEFAULT_SUBTEL_LEN})")
    parser.add_argument("--cen-len", type=int, default=DEFAULT_CEN_LEN,
                        help=f"Centromere length in bp (default: {DEFAULT_CEN_LEN})")
    parser.add_argument("--cen-r-factor", type=float, default=R_CENTROMERE_FACTOR,
                        help=f"Centromere recombination rate as fraction of sub-telomeric rate "
                             f"(default: {R_CENTROMERE_FACTOR})")
    parser.add_argument("--target-snps-per-mb", type=int, default=1000,
                        help="Target SNP density in sub-telomeric regions (default: 1000/Mb)")
    parser.add_argument("--n-reps", type=int, default=1,
                        help="Number of replicate simulations (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--out-prefix", default="sweep_sim",
                        help="Output file prefix (default: sweep_sim)")
    args = parser.parse_args()

    assert args.subtel_len * 2 + args.cen_len == args.chrom_len, (
        f"subtel_len*2 + cen_len ({args.subtel_len*2 + args.cen_len}) "
        f"must equal chrom_len ({args.chrom_len})"
    )

    # --- Load best-fit parameters ---
    alpha, k, ne_mults = get_best_match_params()

    # --- Recombination rates ---
    f_s         = 1.0 / (k + 1.0)
    r_eff       = R_PER_MEIOSIS * f_s
    r_centromere = r_eff * args.cen_r_factor

    print(f"\nRecombination rates:")
    print(f"  Sub-telomeric: {r_eff:.3e} per generation")
    print(f"  Centromere:    {r_centromere:.3e} per generation "
          f"({args.cen_r_factor*100:.1f}% of sub-telomeric)")

    # --- Rate map ---
    rate_map = build_rate_map(
        chrom_len=args.chrom_len,
        subtel_len=args.subtel_len,
        cen_len=args.cen_len,
        r_eff=r_eff,
        r_centromere=r_centromere,
    )

    print(f"\nChromosome structure ({args.chrom_len/1e6:.1f} Mb total):")
    print(f"  0 – {args.subtel_len/1e6:.1f} Mb      sub-telomeric (r = {r_eff:.3e})")
    print(f"  {args.subtel_len/1e6:.1f} – "
          f"{(args.subtel_len+args.cen_len)/1e6:.1f} Mb  centromere    (r = {r_centromere:.3e})")
    print(f"  {(args.subtel_len+args.cen_len)/1e6:.1f} – "
          f"{args.chrom_len/1e6:.1f} Mb  sub-telomeric (r = {r_eff:.3e})")

    # --- Simulate ---
    for rep in range(args.n_reps):
        seed = args.seed + rep if args.seed is not None else None
        suffix = f"_rep{rep+1}" if args.n_reps > 1 else ""
        prefix = f"{args.out_prefix}{suffix}"

        print(f"\nRunning simulation{' rep ' + str(rep+1) if args.n_reps > 1 else ''}...")
        mts = simulate(alpha, ne_mults, rate_map,
                       subtel_len=args.subtel_len,
                       cen_len=args.cen_len,
                       target_snps_per_mb=args.target_snps_per_mb,
                       random_seed=seed)

        write_vcf(mts, f"{prefix}.vcf")
        write_ms(mts, f"{prefix}.ms", args.chrom_len)
        write_region_bed(args.subtel_len, args.cen_len, args.chrom_len,
                         f"{prefix}_regions.bed")

    print("\nDone.")
    print("\nSweeD usage:")
    print(f"  SweeD -name {args.out_prefix} -input {args.out_prefix}.vcf -vcf -grid 1000")
    print(f"  SweeD -name {args.out_prefix} -input {args.out_prefix}.ms -ms -grid 1000")
    print("\nRaisD usage:")
    print(f"  RaiSD -n {args.out_prefix} -I {args.out_prefix}.vcf -f")


if __name__ == "__main__":
    main()
