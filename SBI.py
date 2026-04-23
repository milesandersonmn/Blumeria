import msprime
import numpy as np
import math
import allel
import pandas as pd
import matplotlib.pyplot as plt
import helper_functions

def simulate_beta_demogrpahy(alpha,
                              N0, Ne_windows, breakpoints,
                              n_samples, rate_map, mu, ploidy = 1):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N0)

    # Add Ne change at each time window breakpoint
    for t, Ne in zip(breakpoints, Ne_windows):
        demography.add_population_parameters_change(
            time=t, population="A", initial_size=N0*Ne
        )
    
    ts = msprime.sim_ancestry(
        samples=n_samples,
        demography=demography,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha=alpha),
        ploidy=ploidy  # alpha in (1, 2)
    )
    return ts



#--------------------
#Simulation function
#--------------------
def sim_summary_stats(alpha, k, Ne1, Ne2, Ne3, Ne4, Ne5, Ne6, Ne7, Ne8, Ne9, Ne10, Ne11):
    
    Ne = 40000

    sample_size = 43
    target_min = 2860
    target_max = 2900
   
    mu_per_mitotic_gen  = 5e-7   # mutation rate per mitotic generation
    r_per_meiosis       = 3.37e-7   # recombination rate per sexual generation
    k                   = k     # asexual gens per sexual gen: FREE parameter

    # --- Rescale ---
    f_s = 1 / (k + 1)
    r_eff = r_per_meiosis * f_s  # effective recombination rate per generation

    exclude_ac_below = 2
    ploidy = 1



    r_break = math.log(2) #Recombination rate needed to satisfy probability 2^-t inheritance of two chromosomes
    chrom_positions = [0, 1e6, 2e6, 3e6] #1Mb chromosome sizes
    map_positions = [
        chrom_positions[0],
        chrom_positions[1],
        chrom_positions[1] + 1,
        chrom_positions[2],
        chrom_positions[2] + 1,
        chrom_positions[3]
    ]
    rates = [r_eff, r_break, r_eff, r_break, r_eff] 
    rate_map = msprime.RateMap(position=map_positions, rate=rates) #Rate map for separate chromosomes


    alpha = alpha
    Ne = Ne
    Ne_windows = [Ne1, Ne2, Ne3, Ne4, Ne5, Ne6, Ne7, Ne8, Ne9, Ne10, Ne11]

    import numpy as np

    # Fixed time window boundaries (older boundary of each window, recent to ancient)
    # Windows: 50-10, 100-50, 200-100, 300-200, 400-300, 500-400, 1000-500,
    #          10000-1000, 100000-10000, 1000000-100000, 10000000-1000000
    # N0 covers 0-10 generations
    breakpoints = [10, 50, 100, 200, 300, 400, 500, 1000, 10000, 100000, 1000000]
    print(breakpoints)


    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------

    attempt = 0
    while True:
        attempt += 1
        ts = simulate_beta_demogrpahy(alpha = alpha, N0=Ne, Ne_windows=Ne_windows, breakpoints=breakpoints,
                                      n_samples=sample_size, rate_map=rate_map, ploidy=ploidy, mu=mu_per_mitotic_gen)

        mts = helper_functions.mutation_model(ts = ts, mu = mu_per_mitotic_gen)
        S = mts.num_sites
        
        if S == 0:
            Ne = Ne * 2
            continue
        if target_min > S:
            Ne = (target_min / S) * Ne
        if target_max < S:
            Ne = (target_max / S) * Ne
        if target_min <= S <= target_max:
            #print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne = Ne
            np.set_printoptions(legacy="1.21") #exclude dbtype from np arrays
            summary_statistics = [] #Initialize list of summary statistics

            
            afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=True)

            afs_entries = helper_functions.add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
            

            afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
            summary_statistics.append(afs_quant[0]) #AFS quantile 0.1
            summary_statistics.append(afs_quant[1]) #0.3
            summary_statistics.append(afs_quant[2]) #0.5
            summary_statistics.append(afs_quant[3]) #0.7
            summary_statistics.append(afs_quant[4]) #0.9

            summary_statistics.append(helper_functions.sfs_symmetry_ratio(afs, sample_size)) #SFS symmetry ratio

            num_windows = 30

            D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
            summary_statistics.append(np.nanmean(D_array)) #32 mean Tajima's D
            summary_statistics.append(np.nanvar(D_array)) #33 variance of Tajima's D
            summary_statistics.append(np.nanstd(D_array)) #34 std D
            summary_statistics.append(np.nanstd(D_array) / np.nanmean(D_array)) #35 CV Tajima's D
            
            #split genome into chromosomes
            ts_chroms = []
            for j in range(len(chrom_positions) - 1): 
                start, end = chrom_positions[j: j + 2]
                chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
                ts_chroms.append(chrom_ts)
                #print(chrom_ts.sequence_length)



            s, s_norm, mask = helper_functions.calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below, sample_size = sample_size)
            
            

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            

          
            helper_functions.calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
           
            

            r2 = helper_functions.get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            
            r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])


            summary_statistics.append(r2_quant[0]) #54-58 columns are r^2 quantiles
            summary_statistics.append(r2_quant[1])
            summary_statistics.append(r2_quant[2])
            summary_statistics.append(r2_quant[3])
            summary_statistics.append(r2_quant[4])
            summary_statistics.append(r2_quant[5])
            summary_statistics.append(r2_quant[6])
            summary_statistics.append(np.nanmean(r2)) #59 mean r^2
            summary_statistics.append(np.nanvar(r2)) #60 var r^2
            summary_statistics.append(np.nanstd(r2)) #61 std r^2
            summary_statistics.append(np.nanstd(r2) / np.nanmean(r2)) #62 CV r^2
            summary_statistics.append(np.nanmean(r2) - r2_quant[2]) #mean - median r^2
            summary_statistics.append(r2_ge_1)
           


            ild_all = helper_functions.get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
          
            

            summary_statistics.append(ild_quant[0]) #62-66 ILD quantiles
            summary_statistics.append(ild_quant[1])
            summary_statistics.append(ild_quant[2])
            summary_statistics.append(ild_quant[3])
            summary_statistics.append(ild_quant[4])
            summary_statistics.append(ild_quant[5])
            summary_statistics.append(ild_quant[6])
            summary_statistics.append(np.nanmean(ild_all)) #67 mean ILD
            summary_statistics.append(np.nanvar(ild_all)) #68 var ILD
            summary_statistics.append(np.nanstd(ild_all)) #69 std ILD
            summary_statistics.append(np.nanstd(ild_all) / np.nanmean(ild_all)) #70 CV ILD
            summary_statistics.append(np.nanmean(ild_all) - ild_quant[2]) #mean - median ILD
            summary_statistics.append(ild_ge_1)
            

         
            
            """
            scaled_r2 = helper_functions.calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
            scaled_r2_quant = np.nanquantile(scaled_r2, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])



            summary_statistics.append(scaled_r2_quant[0]) #70-77 Anderson's r^2 quantiles, mean, variance, std
            summary_statistics.append(scaled_r2_quant[1])
            summary_statistics.append(scaled_r2_quant[2])
            summary_statistics.append(scaled_r2_quant[3])
            summary_statistics.append(scaled_r2_quant[4])
            summary_statistics.append(scaled_r2_quant[5])
            summary_statistics.append(scaled_r2_quant[6])
            summary_statistics.append(np.nanmean(scaled_r2))
            summary_statistics.append(np.nanvar(scaled_r2))
            summary_statistics.append(np.nanstd(scaled_r2))
            """


            r2_norm = helper_functions.get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
           
            r2_norm_quant = np.nanquantile(r2_norm, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])


            summary_statistics.append(r2_norm_quant[0]) #78-85 Normalized r^2 quantiles, mean, variance, std
            summary_statistics.append(r2_norm_quant[1])
            summary_statistics.append(r2_norm_quant[2])
            summary_statistics.append(r2_norm_quant[3])
            summary_statistics.append(r2_norm_quant[4])
            summary_statistics.append(r2_norm_quant[5])
            summary_statistics.append(r2_norm_quant[6])
            summary_statistics.append(np.nanmean(r2_norm))
            summary_statistics.append(np.nanvar(r2_norm))
            summary_statistics.append(np.nanstd(r2_norm))
            summary_statistics.append(np.nanstd(r2_norm) / np.nanmean(r2_norm)) #CV r2_norm
            summary_statistics.append(np.nanmean(r2_norm) - r2_norm_quant[2]) #mean - median r2_norm
            summary_statistics.append(r2_norm_ge_1)
                

            ild_norm_all = helper_functions.get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
               
            ild_norm_all_quant = np.nanquantile(ild_norm_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])


            summary_statistics.append(ild_norm_all_quant[0]) #86-93 Normalized ILD r^2 quantiles, mean, variance, std
            summary_statistics.append(ild_norm_all_quant[1])
            summary_statistics.append(ild_norm_all_quant[2])
            summary_statistics.append(ild_norm_all_quant[3])
            summary_statistics.append(ild_norm_all_quant[4])
            summary_statistics.append(ild_norm_all_quant[5])
            summary_statistics.append(ild_norm_all_quant[6])
            summary_statistics.append(np.nanmean(ild_norm_all))
            summary_statistics.append(np.nanvar(ild_norm_all))
            summary_statistics.append(np.nanstd(ild_norm_all))
            summary_statistics.append(np.nanstd(ild_norm_all) / np.nanmean(ild_norm_all)) #CV ild_norm
            summary_statistics.append(np.nanmean(ild_norm_all) - ild_norm_all_quant[2]) #mean - median ild_norm
            summary_statistics.append(ild_norm_ge_1)
           
            """
            scaled_r2_norm = helper_functions.calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
            scaled_r2_norm_quant = np.nanquantile(scaled_r2_norm, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])


            summary_statistics.append(scaled_r2_norm_quant[0]) #94-101 Norm And R-squared r^2 quantiles, mean, variance, std
            summary_statistics.append(scaled_r2_norm_quant[1])
            summary_statistics.append(scaled_r2_norm_quant[2])
            summary_statistics.append(scaled_r2_norm_quant[3])
            summary_statistics.append(scaled_r2_norm_quant[4])
            summary_statistics.append(scaled_r2_norm_quant[5])
            summary_statistics.append(scaled_r2_norm_quant[6])
            summary_statistics.append(np.nanmean(scaled_r2_norm))
            summary_statistics.append(np.nanvar(scaled_r2_norm))
            summary_statistics.append(np.nanstd(scaled_r2_norm))               
            """
            
            samples = mts.samples()
            n = len(samples)

            windows = np.linspace(0, ts.sequence_length, num_windows+1)

            a1 = np.sum(1 / np.arange(1, n))

            theta_pi = []
            theta_w = []

            for left, right in zip(windows[:-1], windows[1:]):

                S = 0
                pi_sum = 0

                for var in mts.variants(samples=samples):
                    if left <= var.site.position < right:

                        # Count alleles among samples
                        alleles, counts = np.unique(var.genotypes, return_counts=True)

                        # Keep only biallelic segregating sites
                        if len(alleles) == 2:
                            S += 1

                            # Compute pi for this site
                            p = counts[1] / n
                            pi_sum += 2 * p * (1 - p)

                theta_pi.append(pi_sum)
                theta_w.append(S / a1)

            theta_pi = np.array(theta_pi)
            theta_w = np.array(theta_w)

            result = np.where(theta_pi > 0,
                            (theta_pi - theta_w) / theta_pi,
                            np.nan)

          
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))
            summary_statistics.append(np.nanstd(result) / np.nanmean(result)) #CV normalized Tajima's D
            
            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
           

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
           
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies


            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
         

            adjacent_r2_stats = helper_functions.get_weighted_rsq_stats_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
 
            summary_statistics.extend([
                adjacent_r2_stats['weighted_quantiles'][0.1],
                adjacent_r2_stats['weighted_quantiles'][0.3],
                adjacent_r2_stats['weighted_quantiles'][0.5],
                adjacent_r2_stats['weighted_quantiles'][0.7],
                adjacent_r2_stats['weighted_quantiles'][0.9],
                adjacent_r2_stats['weighted_mean'],
                adjacent_r2_stats['weighted_std'],
                adjacent_r2_stats['weighted_std'] / adjacent_r2_stats['weighted_mean'],
                adjacent_r2_stats['weighted_mean'] - adjacent_r2_stats['weighted_quantiles'][0.5],

                adjacent_r2_stats['unweighted_quantiles'][0.1],
                adjacent_r2_stats['unweighted_quantiles'][0.3],
                adjacent_r2_stats['unweighted_quantiles'][0.5],
                adjacent_r2_stats['unweighted_quantiles'][0.7],
                adjacent_r2_stats['unweighted_quantiles'][0.9],
                adjacent_r2_stats['unweighted_mean'],
                adjacent_r2_stats['unweighted_std'],
                adjacent_r2_stats['unweighted_std'] / adjacent_r2_stats['unweighted_mean'],
                adjacent_r2_stats['unweighted_mean'] - adjacent_r2_stats['unweighted_quantiles'][0.5],

            ])


            adjacent_norm_r2_stats = helper_functions.get_weighted_rsq_stats_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s_norm)
 
            summary_statistics.extend([
                adjacent_norm_r2_stats['weighted_quantiles'][0.1],
                adjacent_norm_r2_stats['weighted_quantiles'][0.3],
                adjacent_norm_r2_stats['weighted_quantiles'][0.5],
                adjacent_norm_r2_stats['weighted_quantiles'][0.7],
                adjacent_norm_r2_stats['weighted_quantiles'][0.9],
                adjacent_norm_r2_stats['weighted_mean'],
                adjacent_norm_r2_stats['weighted_std'],
                adjacent_norm_r2_stats['weighted_std'] / adjacent_norm_r2_stats['weighted_mean'],
                adjacent_norm_r2_stats['weighted_mean'] - adjacent_norm_r2_stats['weighted_quantiles'][0.5],

                adjacent_norm_r2_stats['unweighted_quantiles'][0.1],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.3],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.5],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.7],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.9],
                adjacent_norm_r2_stats['unweighted_mean'],
                adjacent_norm_r2_stats['unweighted_std'],
                adjacent_norm_r2_stats['unweighted_std'] / adjacent_norm_r2_stats['unweighted_mean'],
                adjacent_norm_r2_stats['unweighted_mean'] - adjacent_norm_r2_stats['unweighted_quantiles'][0.5],
            ])
            print(summary_statistics)

            """
            window_size = 100_000
            genome_length = int(mts.sequence_length)
            ic = 2  # singletons vs non-singletons; adjust as needed

            results = []
            for window_start in range(0, genome_length, window_size):
                window_end = window_start + window_size
                hilo_PMI, eta_hilo, eta_lo, eta_hi  = helper_functions.compute_window_hiloPMI(mts, n, window_start, window_end, ic=ic)
                results.append({
                    "window_start": window_start,
                    "window_end":   window_end,
                    "hilo_PMI":     hilo_PMI,
                    "eta_hilo":     eta_hilo,
                    "eta_lo":       eta_lo,
                    "eta_hi":       eta_hi
                })
                print(f"[{window_start:>10} - {window_end:>10}] hilo_PMI = {hilo_PMI}, eta_hilo = {eta_hilo}, eta_lo = {eta_lo}, eta_hi = {eta_hi}")
            valid_results = [r for r in results if r["eta_hilo"] is not None]

            means = {key: np.mean([r[key] for r in valid_results]) for key in valid_results[0]}
            print(means)

            print(np.log(means["eta_hilo"]/(means["eta_lo"] * means["eta_hi"])))
            print("norm Taj D:",np.nanmean(result))
            print((np.log(means["eta_hilo"]/(means["eta_lo"] * means["eta_hi"])))/np.nanmean(result))

            hiloPMI = np.log(means["eta_hilo"]/(means["eta_lo"] * means["eta_hi"]))
            summary_statistics.append(hiloPMI) #148 HiloPMI

            norm_hiloPMI = np.log(means["eta_hilo"]/(means["eta_lo"] * means["eta_hi"]))/np.nanmean(result)
            summary_statistics.append(norm_hiloPMI) #149 Tajima's D normed HiloPMI

            print("hiloPMI", [d["hilo_PMI"] for d in results])
            print("norm D", result)
            print(np.nanmean([d["hilo_PMI"] for d in results]/result))
            print(np.nanmean([d["hilo_PMI"] for d in results]/result)**2)
            window_hiloPMI_sq = np.nanmean([d["hilo_PMI"] for d in results]/result)**2
            summary_statistics.append(window_hiloPMI_sq) #150 (HiloPMI per window normalized by norm_D)**2
            #print((np.nanmean([d["hilo_PMI"] for d in results]/result)/np.nanstd([d["hilo_PMI"] for d in results]/result))**2)
            """
    



            return summary_statistics
           

from sbi import inference as sbi_inference
from sbi.utils import BoxUniform
import torch
import numpy as np
import os
import sys
from contextlib import contextmanager

@contextmanager
def stdout_to_log(log_path):
    """Redirect stdout to a log file, leaving stderr (tqdm) untouched."""
    with open(log_path, "a") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor

# Define priors
prior = BoxUniform(
    #low=torch.tensor([1.3, 10.0, 0.001, 10.0]),   # alpha, t_b, severity, duration
    #high=torch.tensor([1.9, 100000.0, 1.0, 5000.0])
    low=torch.tensor([1.3, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),   # alpha, k, Ne1-Ne11
    high=torch.tensor([1.9, 150, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
)

def simulator(params):
    params = params.numpy()
    if params.ndim == 2:
        params = params[0]
    params = params.flatten()
    alpha, k, Ne1, Ne2, Ne3, Ne4, Ne5, Ne6, Ne7, Ne8, Ne9, Ne10, Ne11 = params
    stats = sim_summary_stats(alpha, k, Ne1, Ne2, Ne3, Ne4, Ne5, Ne6, Ne7, Ne8, Ne9, Ne10, Ne11)
    return torch.tensor(np.array(stats), dtype=torch.float32).flatten()

def run_single_sim(_):
    with stdout_to_log("sim.log"):
        params = prior.sample((1,)).squeeze()
        stats = simulator(params)
    return params, stats


def run_ppc_sim(params_np):
    try:
        with stdout_to_log("sim.log"):
            p = torch.tensor(params_np, dtype=torch.float32)
            return simulator(p).numpy()
    except Exception:
        return None


def run_sbc_sim(_):
    """Sample theta* from prior, simulate x*, return (theta*, x*) as numpy arrays."""
    try:
        with stdout_to_log("sim.log"):
            theta_star = prior.sample((1,)).squeeze()
            x_star = simulator(theta_star)
        return theta_star.numpy(), x_star.numpy()
    except Exception:
        return None


# ---- SFS confounding experiment helpers (top-level for pickling) ----

def _confound_sim_worker(args):
    """Single simulation for the SFS confounding grid."""
    alpha, ne_recent = args
    with stdout_to_log("sim.log"):
        # ne_recent applied to Ne1-Ne7 (10-1000 gen); Ne8-Ne11 held flat at 1.0
        stats = sim_summary_stats(alpha, 1,
                                  ne_recent, ne_recent, ne_recent, ne_recent,
                                  ne_recent, ne_recent, ne_recent,
                                  1, 1, 1, 1)
    return np.array(stats[:42])  # SFS bins 1-42


def run_sfs_confounding_experiment(num_workers, n_sims=20):
    """
    Simulate SFS under four conditions demonstrating confounding between
    alpha (Beta coalescent) and recent demography (Ne1-Ne7, 10-1000 gen).

    Conditions
    ----------
    1. Low alpha (1.4), flat demography     — pure low-alpha signal
    2. High alpha (1.8), flat demography    — pure high-alpha signal
    3. High alpha (1.8) + recent expansion  — mimics low alpha (excess singletons)
    4. Low alpha (1.4) + recent contraction — mimics high alpha (excess intermediates)
    """
    # Okabe-Ito colorblind-safe palette (blue, vermillion, bluish-green, orange)
    conditions = [
        ("Low \u03b1 (1.4), flat",            1.4, 1.0, "#0072B2", "-",  "o"),
        ("High \u03b1 (1.8), flat",           1.8, 1.0, "#D55E00", "--", "s"),
        ("High \u03b1 (1.8) + expansion",     1.8, 0.1, "#009E73", "-.", "^"),
        ("Low \u03b1 (1.4) + contraction",    1.4, 5.0, "#E69F00", ":",  "D"),
    ]
    bin_labels = np.arange(1, 43)

    # Collect results first, then plot
    results_by_cond = []
    for (label, alpha, ne_recent, color, ls, marker) in conditions:
        job_args = [(alpha, ne_recent)] * n_sims
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            res = list(tqdm(executor.map(_confound_sim_worker, job_args),
                            total=n_sims, desc=label))
        mat = np.array(res)
        results_by_cond.append((label, color, ls, marker, mat.mean(axis=0), mat.std(axis=0)))

    # Two-panel plot: baselines (left) vs confounded (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel_indices = [[0, 1], [2, 3]]   # which conditions go in each panel
    panel_titles = [
        "Baseline (flat demography)",
        "Confounded (recent size change)",
    ]
    # Plot baselines in both panels as light reference lines
    for ax, idx_list, title in zip(axes, panel_indices, panel_titles):
        # faint reference: the two baselines always shown
        for ri in [0, 1]:
            label, color, ls, marker, mean_sfs, std_sfs = results_by_cond[ri]
            lw = 2.5 if ri in idx_list else 1.0
            al = 1.0 if ri in idx_list else 0.3
            ax.plot(bin_labels, mean_sfs, label=label if ri in idx_list else None,
                    color=color, linestyle=ls, linewidth=lw, alpha=al,
                    marker=marker, markevery=6, markersize=5)
            if ri in idx_list:
                ax.fill_between(bin_labels,
                                mean_sfs - std_sfs,
                                mean_sfs + std_sfs,
                                alpha=0.15, color=color)
        # the confounded condition(s) for this panel
        for ri in idx_list:
            if ri in [0, 1]:
                continue
            label, color, ls, marker, mean_sfs, std_sfs = results_by_cond[ri]
            ax.plot(bin_labels, mean_sfs, label=label,
                    color=color, linestyle=ls, linewidth=2.5,
                    marker=marker, markevery=6, markersize=5)
            ax.fill_between(bin_labels,
                            mean_sfs - std_sfs,
                            mean_sfs + std_sfs,
                            alpha=0.15, color=color)
        ax.set_xlabel("Derived allele count", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)

    axes[0].set_ylabel("Proportion of segregating sites", fontsize=11)
    fig.suptitle(
        "SFS confounding: \u03b1 (Beta coalescent) vs. recent demography (Ne1\u2013Ne7, 10\u20131000 gen)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig("sfs_confounding.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved sfs_confounding.png")


def plot_ppc(ppc_stats, x_obs_np, out_path="posterior_predictive_check.pdf"):
    """Save all summary statistics to a multi-page PDF, 20 panels per page."""
    from matplotlib.backends.backend_pdf import PdfPages
    n_stats = ppc_stats.shape[1]
    per_page = 20
    ncols, nrows = 5, 4
    n_pages = math.ceil(n_stats / per_page)

    with PdfPages(out_path) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
            for slot, ax in enumerate(axes.flat):
                k = page * per_page + slot
                if k >= n_stats:
                    ax.axis("off")
                    continue
                ax.hist(ppc_stats[:, k], bins=30, density=True,
                        color="steelblue", alpha=0.7, label="PPC")
                ax.axvline(x_obs_np[k], color="red", linewidth=2, label="Observed")
                label = STAT_NAMES[k] if k < len(STAT_NAMES) else f"Stat {k}"
                ax.set_title(label, fontsize=7)
                if k == 0:
                    ax.legend(fontsize=8)
            plt.suptitle(
                f"Posterior predictive check — stats {page*per_page+1}–{min((page+1)*per_page, n_stats)} of {n_stats}",
                y=1.01, fontsize=10
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {out_path} ({n_pages} pages, {n_stats} statistics)")


def run_shap_analysis(x_np, theta_np, rf_models, param_names, num_workers=1):
    import shap

    background_size = min(500, x_np.shape[0])
    rng_shap = np.random.default_rng(0)
    bg_idx = rng_shap.choice(x_np.shape[0], size=background_size, replace=False)
    X_bg = x_np[bg_idx]

    sfs_bin1_idx = 0
    ne_recent_bg = theta_np[bg_idx, 2:9].mean(axis=1)
    singleton_bg = X_bg[:, sfs_bin1_idx]

    print(f"Computing SHAP values for {len(param_names)} parameters...")
    shap_results = []
    for name in tqdm(param_names, desc="SHAP"):
        explainer = shap.TreeExplainer(rf_models[name])
        shap_results.append(explainer.shap_values(X_bg))

    ncols = 3
    nrows = math.ceil(len(param_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for pi, (name, shap_vals) in enumerate(zip(param_names, shap_results)):
        ax = axes[pi // ncols][pi % ncols]
        shap_for_bin1 = shap_vals[:, sfs_bin1_idx]
        sc = ax.scatter(ne_recent_bg, shap_for_bin1,
                        c=singleton_bg, cmap="RdBu_r",
                        s=12, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("SFS bin 1 value", fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean Ne1–Ne7 (recent Ne multiplier)", fontsize=9)
        ax.set_ylabel(f"SHAP value of SFS bin 1\nfor RF({name})", fontsize=8)
        ax.set_title(f"Parameter: {name}", fontsize=10)

    for pi in range(len(param_names), nrows * ncols):
        axes[pi // ncols][pi % ncols].axis("off")

    plt.suptitle(
        "SHAP: how SFS singleton bin 1 drives each parameter's RF prediction\n"
        "as a function of recent Ne (mean Ne1–Ne7) — confounding diagnostic",
        y=1.01, fontsize=11
    )
    plt.tight_layout()
    plt.savefig("shap_alpha_Ne_interaction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved shap_alpha_Ne_interaction.png")


# ---- Summary statistic names (195 total, matches x.shape[1]) ----
STAT_NAMES = (
    [f"SFS bin {i}" for i in range(1, 43)]          # 0-41
    + [f"AFS q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 42-46
    + ["SFS symmetry ratio"]                          # 47
    + ["Tajima's D mean", "Tajima's D var", "Tajima's D std", "Tajima's D CV"]  # 48-51
    + [f"Hamming q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 52-56
    + ["Hamming mean", "Hamming std", "Hamming var"]  # 57-59
    + [f"r² q{q}" for q in ["0.1","0.3","0.5","0.7","0.9","0.95","0.99"]]  # 60-66
    + ["r² mean", "r² var", "r² std", "r² CV", "r² mean-median", "r²≥1 prop"]  # 67-72
    + [f"ILD q{q}" for q in ["0.1","0.3","0.5","0.7","0.9","0.95","0.99"]]  # 73-79
    + ["ILD mean", "ILD var", "ILD std", "ILD CV", "ILD mean-median", "ILD≥1 prop"]  # 80-85
    + [f"r²_norm q{q}" for q in ["0.1","0.3","0.5","0.7","0.9","0.95","0.99"]]  # 86-92
    + ["r²_norm mean", "r²_norm var", "r²_norm std", "r²_norm CV", "r²_norm mean-median", "r²_norm≥1 prop"]  # 93-98
    + [f"ILD_norm q{q}" for q in ["0.1","0.3","0.5","0.7","0.9","0.95","0.99"]]  # 99-105
    + ["ILD_norm mean", "ILD_norm var", "ILD_norm std", "ILD_norm CV", "ILD_norm mean-median", "ILD_norm≥1 prop"]  # 106-111
    + ["norm Taj.D mean", "norm Taj.D std", "norm Taj.D CV"]  # 112-114
    + [f"r² LD-spec [{0.1*i:.1f}-{0.1*(i+1):.1f}]" for i in range(10)]  # 115-124
    + ["r² LD-spec diff"]  # 125
    + [f"ILD LD-spec [{0.1*i:.1f}-{0.1*(i+1):.1f}]" for i in range(10)]  # 126-135
    + ["ILD LD-spec diff"]  # 136
    + [f"r²_norm LD-spec [{0.1*i:.1f}-{0.1*(i+1):.1f}]" for i in range(10)]  # 137-146
    + ["r²_norm LD-spec diff"]  # 147
    + [f"ILD_norm LD-spec [{0.1*i:.1f}-{0.1*(i+1):.1f}]" for i in range(10)]  # 148-157
    + ["ILD_norm LD-spec diff"]  # 158
    + [f"adj-r² wtd q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 159-163
    + ["adj-r² wtd mean", "adj-r² wtd std", "adj-r² wtd CV", "adj-r² wtd mean-med"]  # 164-167
    + [f"adj-r² unwtd q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 168-172
    + ["adj-r² unwtd mean", "adj-r² unwtd std", "adj-r² unwtd CV", "adj-r² unwtd mean-med"]  # 173-176
    + [f"adj-r²_norm wtd q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 177-181
    + ["adj-r²_norm wtd mean", "adj-r²_norm wtd std", "adj-r²_norm wtd CV", "adj-r²_norm wtd mean-med"]  # 182-185
    + [f"adj-r²_norm unwtd q{q}" for q in ["0.1","0.3","0.5","0.7","0.9"]]  # 186-190
    + ["adj-r²_norm unwtd mean", "adj-r²_norm unwtd std", "adj-r²_norm unwtd CV", "adj-r²_norm unwtd mean-med"]  # 191-194
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--confound-only", action="store_true",
                        help="Run only the SFS confounding experiment and exit")
    parser.add_argument("--ppc-only", action="store_true",
                        help="Load saved simulations, retrain NPE, run PPC only, then exit")
    parser.add_argument("--shap-only", action="store_true",
                        help="Load saved simulations, train RF, run SHAP analysis only, then exit")
    parser.add_argument("--sbc-only", action="store_true",
                        help="Run simulation-based calibration and exit")
    parser.add_argument("--sbc-trials", type=int, default=500,
                        help="Number of SBC trials (default: 500)")
    parser.add_argument("--sbc-samples", type=int, default=100,
                        help="Posterior samples per SBC trial (default: 100)")
    parser.add_argument("--nsf", action="store_true",
                        help="Retrain using a neural spline flow and save as posterior_nsf.pt, then exit")
    parser.add_argument("--posterior-file", type=str, default="posterior.pt",
                        help="Posterior file to use for --ppc-only, --sbc-only, and --posterior-plots (default: posterior.pt)")
    parser.add_argument("--posterior-plots", action="store_true",
                        help="Load posterior and produce marginal + joint posterior plots, then exit")
    args = parser.parse_args()

    num_workers = os.cpu_count() - 1

    if args.confound_only:
        run_sfs_confounding_experiment(num_workers=num_workers, n_sims=20)
        import sys; sys.exit(0)

    if args.ppc_only:
        import sys
        param_names = ["alpha", "k", "Ne1", "Ne2", "Ne3", "Ne4", "Ne5", "Ne6", "Ne7", "Ne8", "Ne9", "Ne10", "Ne11"]
        posterior_path = args.posterior_file
        if os.path.exists(posterior_path):
            print(f"Loading saved posterior from {posterior_path}...")
            posterior = torch.load(posterior_path)
        else:
            print(f"{posterior_path} not found — loading simulations and retraining NPE...")
            theta = torch.tensor(np.load("theta.npy"), dtype=torch.float32)
            x     = torch.tensor(np.load("x.npy"),     dtype=torch.float32)
            print(f"Loaded theta {theta.shape}, x {x.shape}")
            inferrer = sbi_inference.SNPE(prior=prior)
            inferrer.append_simulations(theta, x)
            density_estimator = inferrer.train()
            posterior = inferrer.build_posterior(density_estimator)
            torch.save(posterior, "posterior.pt")
            print("Saved posterior.pt")

        x_obs = torch.tensor(pd.read_csv("observed_sum_stats_SBI.csv").values, dtype=torch.float32).squeeze()

        n_ppc = 500
        ppc_params = posterior.sample((n_ppc,), x=x_obs)
        ppc_params_list = [ppc_params[j].numpy() for j in range(n_ppc)]

        print(f"Running {n_ppc} PPC simulations across {num_workers} cores...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            ppc_results = list(tqdm(executor.map(run_ppc_sim, ppc_params_list),
                                    total=n_ppc, desc="PPC simulations"))

        ppc_stats = np.array([r for r in ppc_results if r is not None])
        x_obs_np = x_obs.numpy()

        plot_ppc(ppc_stats, x_obs_np)
        sys.exit(0)

    if args.shap_only:
        import sys
        from sklearn.ensemble import RandomForestRegressor
        param_names = ["alpha", "k", "Ne1", "Ne2", "Ne3", "Ne4", "Ne5", "Ne6", "Ne7", "Ne8", "Ne9", "Ne10", "Ne11"]
        print("Loading saved simulations...")
        x_np    = np.load("x.npy")
        theta_np = np.load("theta.npy")
        print(f"Loaded theta {theta_np.shape}, x {x_np.shape}")
        import joblib
        rf_path = "rf_models.pkl"
        if os.path.exists(rf_path):
            print("Loading saved random forests...")
            rf_models = joblib.load(rf_path)
        else:
            print("Training random forests...")
            rf_models = {}
            for i, name in enumerate(param_names):
                rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
                rf.fit(x_np, theta_np[:, i])
                rf_models[name] = rf
                print(f"  {name} done")
            joblib.dump(rf_models, rf_path)
            print(f"Saved {rf_path}")
        run_shap_analysis(x_np, theta_np, rf_models, param_names)
        sys.exit(0)

    if args.nsf:
        import sys
        from sbi.neural_nets import posterior_nn
        print("Loading saved simulations...")
        theta = torch.tensor(np.load("theta.npy"), dtype=torch.float32)
        x     = torch.tensor(np.load("x.npy"),     dtype=torch.float32)
        print(f"Loaded theta {theta.shape}, x {x.shape}")
        print("Training neural spline flow...")
        density_estimator_build_fun = posterior_nn(
            model="nsf",
            hidden_features=64,
            num_transforms=5,
        )
        inferrer = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator_build_fun)
        inferrer.append_simulations(theta, x)
        density_estimator = inferrer.train()
        posterior_nsf = inferrer.build_posterior(density_estimator)
        torch.save(posterior_nsf, "posterior_nsf.pt")
        print("Saved posterior_nsf.pt")
        sys.exit(0)

    if args.posterior_plots:
        import sys
        param_names = ["alpha", "k", "Ne1", "Ne2", "Ne3", "Ne4", "Ne5", "Ne6", "Ne7", "Ne8", "Ne9", "Ne10", "Ne11"]
        posterior_path = args.posterior_file
        stem = os.path.splitext(os.path.basename(posterior_path))[0]  # e.g. "posterior_nsf"

        print(f"Loading posterior from {posterior_path}...")
        posterior = torch.load(posterior_path)
        x_obs = torch.tensor(pd.read_csv("observed_sum_stats_SBI.csv").values, dtype=torch.float32).squeeze()

        print("Sampling posterior...")
        samples = posterior.sample((10_000,), x=x_obs)
        samples_np = samples.numpy()

        for i, name in enumerate(param_names):
            print(f"{name}: mean={samples_np[:, i].mean():.4f}, "
                  f"std={samples_np[:, i].std():.4f}, "
                  f"95% CI=[{np.percentile(samples_np[:, i], 2.5):.4f}, "
                  f"{np.percentile(samples_np[:, i], 97.5):.4f}]")

        pd.DataFrame(samples_np, columns=param_names).to_csv(f"{stem}_samples.csv", index=False)
        print(f"Saved {stem}_samples.csv")

        # Marginal posteriors
        fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 5))
        for i, (name, ax) in enumerate(zip(param_names, axes)):
            col = samples_np[:, i]
            mean = col.mean()
            std = col.std()
            ci_lo = np.percentile(col, 2.5)
            ci_hi = np.percentile(col, 97.5)
            ax.hist(col, bins=50, density=True, color="steelblue", alpha=0.7)
            ax.axvline(mean, color="black", linewidth=1.2, linestyle="--", label="mean")
            ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="black", label="95% CI")
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.text(0.97, 0.97,
                    f"mean={mean:.3f}\nstd={std:.3f}\nCI=[{ci_lo:.3f}, {ci_hi:.3f}]",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        plt.tight_layout()
        plt.savefig(f"{stem}_marginals.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {stem}_marginals.png")

        # Joint posterior
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples_np[:, i], bins=50, density=True, color="steelblue", alpha=0.7)
                elif i > j:
                    ax.hist2d(samples_np[:, j], samples_np[:, i],
                              bins=50, norm=LogNorm(), cmap="Blues")
                else:
                    ax.axis("off")
                if j == 0:
                    ax.set_ylabel(param_names[i])
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j])
        plt.tight_layout()
        plt.savefig(f"{stem}_joint.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {stem}_joint.png")

        # Pairplot (posterior vs prior)
        prior_samples_np = prior.sample((10_000,)).numpy()
        fig, axes = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                if i == j:
                    ax.hist(prior_samples_np[:, i], bins=50, density=True,
                            color="grey", alpha=0.4, label="Prior")
                    ax.hist(samples_np[:, i], bins=50, density=True,
                            color="steelblue", alpha=0.7, label="Posterior")
                    if i == 0:
                        ax.legend(fontsize=7)
                elif i > j:
                    ax.hist2d(samples_np[:, j], samples_np[:, i],
                              bins=40, norm=LogNorm(), cmap="Blues")
                else:
                    ax.axis("off")
                if j == 0:
                    ax.set_ylabel(param_names[i], fontsize=8)
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j], fontsize=8)
                ax.tick_params(labelsize=6)
        plt.suptitle(f"Posterior pairplot — {stem}", y=1.002)
        plt.tight_layout()
        plt.savefig(f"{stem}_pairplot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {stem}_pairplot.png")
        sys.exit(0)

    if args.sbc_only:
        import sys
        param_names = ["alpha", "k", "Ne1", "Ne2", "Ne3", "Ne4", "Ne5", "Ne6", "Ne7", "Ne8", "Ne9", "Ne10", "Ne11"]
        n_trials  = args.sbc_trials
        L         = args.sbc_samples

        # Load posterior
        posterior_path = args.posterior_file
        if os.path.exists(posterior_path):
            print(f"Loading saved posterior from {posterior_path}...")
            posterior = torch.load(posterior_path)
        else:
            print(f"{posterior_path} not found — loading simulations and retraining NPE...")
            theta = torch.tensor(np.load("theta.npy"), dtype=torch.float32)
            x     = torch.tensor(np.load("x.npy"),     dtype=torch.float32)
            inferrer = sbi_inference.SNPE(prior=prior)
            inferrer.append_simulations(theta, x)
            density_estimator = inferrer.train()
            posterior = inferrer.build_posterior(density_estimator)
            torch.save(posterior, "posterior.pt")
            print("Saved posterior.pt")

        # Run SBC trials in parallel (simulate theta*, x*)
        print(f"Running {n_trials} SBC trials across {num_workers} cores...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            sbc_results = list(tqdm(executor.map(run_sbc_sim, range(n_trials)),
                                    total=n_trials, desc="SBC simulations"))

        sbc_results = [r for r in sbc_results if r is not None]
        print(f"{len(sbc_results)} successful trials")

        # Compute ranks
        ranks = {name: [] for name in param_names}
        for theta_star_np, x_star_np in tqdm(sbc_results, desc="Computing ranks"):
            x_star = torch.tensor(x_star_np, dtype=torch.float32)
            theta_star = torch.tensor(theta_star_np, dtype=torch.float32)
            post_samples = posterior.sample((L,), x=x_star).numpy()
            for i, name in enumerate(param_names):
                rank = int((post_samples[:, i] < theta_star_np[i]).sum())
                ranks[name].append(rank)

        # Save ranks
        pd.DataFrame(ranks).to_csv("sbc_ranks.csv", index=False)
        print("Saved sbc_ranks.csv")

        # Plot rank histograms
        n_params = len(param_names)
        ncols = 4
        nrows = math.ceil(n_params / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        expected = len(sbc_results) / (L + 1)  # expected count per bin if uniform

        for pi, (name, ax) in enumerate(zip(param_names, axes.flat)):
            ax.hist(ranks[name], bins=10, range=(0, L),
                    color='#0072B2', alpha=0.8, edgecolor='white')
            ax.axhline(expected, color='#D55E00', linewidth=1.5,
                       linestyle='--', label='Uniform expected')
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Rank", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            if pi == 0:
                ax.legend(fontsize=7)

        for pi in range(n_params, nrows * ncols):
            axes.flat[pi].axis("off")

        plt.suptitle(
            f"Simulation-based calibration ({len(sbc_results)} trials, {L} posterior samples each)\n"
            "Uniform rank histogram = well-calibrated posterior",
            fontsize=11, y=1.02
        )
        plt.tight_layout()
        plt.savefig("sbc_ranks.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved sbc_ranks.png")
        sys.exit(0)

    # Run new simulations in parallel
    num_new_simulations = 100000
    print(f"Running {num_new_simulations} simulations across {num_workers} cores...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(run_single_sim, range(num_new_simulations)),
                            total=num_new_simulations, desc="Simulations"))

    thetas_new, xs_new = zip(*results)
    theta_new = torch.stack(thetas_new)
    x_new = torch.stack(xs_new)

    # Load and concatenate with previous simulations if they exist and dimensions match
    if os.path.exists("theta.npy") and os.path.exists("x.npy"):
        theta_old = torch.tensor(np.load("theta.npy"), dtype=torch.float32)
        x_old     = torch.tensor(np.load("x.npy"),     dtype=torch.float32)
        if x_old.shape[1] == x_new.shape[1] and theta_old.shape[1] == theta_new.shape[1]:
            theta = torch.cat([theta_old, theta_new], dim=0)
            x     = torch.cat([x_old,     x_new],     dim=0)
            print(f"Loaded {theta_old.shape[0]} previous simulations, "
                  f"total now: {theta.shape[0]}")
        else:
            theta, x = theta_new, x_new
            print(f"Dimension mismatch: old x has {x_old.shape[1]} stats, "
                  f"new x has {x_new.shape[1]}. Discarding old simulations.")
    else:
        theta, x = theta_new, x_new
        print("No previous simulations found, starting fresh.")

    print(f"Final theta shape: {theta.shape}")
    print(f"Final x shape: {x.shape}")

    # Overwrite saved simulations with the full combined set
    np.save("theta.npy", theta.numpy())
    np.save("x.npy",     x.numpy())

    # Train NPE on the full combined set
    inferrer = sbi_inference.SNPE(prior=prior)
    inferrer.append_simulations(theta, x)
    density_estimator = inferrer.train()
    
    """
    # Test simulator output
    test_params = prior.sample((1,)).squeeze()
    test_out = simulator(test_params)
    print(f"shape: {test_out.shape}")
    print(f"ndim: {test_out.ndim}")

    # Run simulations in parallel
    num_simulations = 100000
    num_workers = os.cpu_count() - 1
    print(f"Running {num_simulations} simulations across {num_workers} cores...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_single_sim, range(num_simulations)))

    thetas, xs = zip(*results)
    theta = torch.stack(thetas)
    x = torch.stack(xs)

    print(f"Final theta shape: {theta.shape}")
    print(f"Final x shape: {x.shape}")

    # Save simulations to disk
    np.save("theta.npy", theta.numpy())
    np.save("x.npy", x.numpy())

    # Train NPE
    inferrer = sbi_inference.SNPE(prior=prior)
    inferrer.append_simulations(theta, x)
    density_estimator = inferrer.train()
    """
    posterior = inferrer.build_posterior(density_estimator)
    torch.save(posterior, "posterior.pt")
    print("Saved posterior.pt")
    

    # Sample posterior given observed stats
    x_obs = torch.tensor(pd.read_csv("observed_sum_stats_SBI.csv").values, dtype=torch.float32).squeeze()
    samples = posterior.sample((10_000,), x=x_obs)
    samples_np = samples.numpy()
    param_names = ["alpha", "k", "Ne1", "Ne2", "Ne3", "Ne4", "Ne5", "Ne6", "Ne7", "Ne8", "Ne9", "Ne10", "Ne11"]

    # Print posterior summaries
    for i, name in enumerate(param_names):
        print(f"{name}: mean={samples_np[:, i].mean():.4f}, "
              f"std={samples_np[:, i].std():.4f}, "
              f"95% CI=[{np.percentile(samples_np[:, i], 2.5):.4f}, "
              f"{np.percentile(samples_np[:, i], 97.5):.4f}]")

    # Save posterior samples
    pd.DataFrame(samples_np, columns=param_names).to_csv("posterior_samples.csv", index=False)
    print("Saved posterior_samples.csv")

    # Plot marginal posteriors
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
    for i, (name, ax) in enumerate(zip(param_names, axes)):
        ax.hist(samples_np[:, i], bins=50, density=True, color="steelblue", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("posterior_marginals.png", dpi=150)
    plt.show()
    print("Saved posterior_marginals.png")

    # Plot joint pairwise posteriors
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples_np[:, i], bins=50, density=True,
                        color="steelblue", alpha=0.7)
            elif i > j:
                ax.hist2d(samples_np[:, j], samples_np[:, i],
                          bins=50, norm=LogNorm(), cmap="Blues")
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(param_names[i])
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
    plt.tight_layout()
    plt.savefig("joint_posterior.png", dpi=150)
    plt.show()
    print("Saved joint_posterior.png")
    
    
    # ----------------------------------------
    # RANDOM FOREST SUMMARY STATISTIC IMPORTANCE
    # ----------------------------------------
    from sklearn.ensemble import RandomForestRegressor

    x_np = x.numpy()
    theta_np = theta.numpy()
    n_stats = x_np.shape[1]
    
    rf_models = {}
    importance_matrix = np.zeros((len(param_names), n_stats))
    for i, name in enumerate(param_names):
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
        rf.fit(x_np, theta_np[:, i])
        rf_models[name] = rf
        importance_matrix[i] = rf.feature_importances_
        top_idx = np.argsort(rf.feature_importances_)[::-1][:10]
        print(f"{name} top 10 stats: indices={top_idx}, importances={rf.feature_importances_[top_idx].round(4)}")

    # Plot heatmap of importances
    fig, ax = plt.subplots(figsize=(max(12, n_stats // 4), len(param_names)))
    im = ax.imshow(importance_matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Summary statistic index")
    ax.set_ylabel("Parameter")
    ax.set_title("Random forest feature importance")
    plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()
    plt.savefig("rf_importance.png", dpi=150)
    plt.show()
    print("Saved rf_importance.png")
    
    # ----------------------------------------
    # SFS CONFOUNDING EXPERIMENT
    # ----------------------------------------
    run_sfs_confounding_experiment(num_workers=num_workers, n_sims=20)

    # ----------------------------------------
    # SHAP: ALPHA vs RECENT DEMOGRAPHY INTERACTION
    # ----------------------------------------
    run_shap_analysis(x_np, theta_np, rf_models, param_names)

    # ----------------------------------------
    # POSTERIOR PREDICTIVE CHECK
    # ----------------------------------------
    n_ppc = 500
    ppc_params = posterior.sample((n_ppc,), x=x_obs)
    ppc_params_list = [ppc_params[j].numpy() for j in range(n_ppc)]

    print(f"Running {n_ppc} PPC simulations across {num_workers} cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        ppc_results = list(tqdm(executor.map(run_ppc_sim, ppc_params_list),
                                total=n_ppc, desc="PPC simulations"))

    ppc_stats = np.array([r for r in ppc_results if r is not None])  # shape (n_valid, n_stats)
    x_obs_np = x_obs.numpy()

    plot_ppc(ppc_stats, x_obs_np)

    # ----------------------------------------
    # PAIRPLOT (posterior vs prior)
    # ----------------------------------------
    prior_samples_np = prior.sample((10_000,)).numpy()

    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(prior_samples_np[:, i], bins=50, density=True,
                        color="grey", alpha=0.4, label="Prior")
                ax.hist(samples_np[:, i], bins=50, density=True,
                        color="steelblue", alpha=0.7, label="Posterior")
                if i == 0:
                    ax.legend(fontsize=7)
            elif i > j:
                ax.hist2d(samples_np[:, j], samples_np[:, i],
                          bins=40, norm=LogNorm(), cmap="Blues")
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=8)
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=8)
            ax.tick_params(labelsize=6)
    plt.suptitle("Posterior pairplot (with prior overlay on diagonal)", y=1.002)
    plt.tight_layout()
    plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved pairplot.png")