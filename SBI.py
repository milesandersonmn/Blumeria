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
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
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
    params = prior.sample((1,)).squeeze()
    stats = simulator(params)
    return params, stats


# ---- SFS confounding experiment helpers (top-level for pickling) ----

def _confound_sim_worker(args):
    """Single simulation for the SFS confounding grid."""
    alpha, ne_recent = args
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
    conditions = [
        ("Low alpha (1.4), flat demography",        1.4, 1.0),
        ("High alpha (1.8), flat demography",       1.8, 1.0),
        ("High alpha (1.8) + recent expansion",     1.8, 0.1),
        ("Low alpha (1.4) + recent contraction",    1.4, 5.0),
    ]
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    bin_labels = np.arange(1, 43)

    fig, ax = plt.subplots(figsize=(14, 6))

    for (label, alpha, ne_recent), color in zip(conditions, colors):
        job_args = [(alpha, ne_recent)] * n_sims
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_confound_sim_worker, job_args),
                                total=n_sims, desc=label))

        mat = np.array(results)
        mean_sfs = mat.mean(axis=0)
        std_sfs = mat.std(axis=0)

        ax.plot(bin_labels, mean_sfs, label=label, color=color, linewidth=2)
        ax.fill_between(bin_labels,
                        mean_sfs - std_sfs,
                        mean_sfs + std_sfs,
                        alpha=0.2, color=color)

    ax.set_xlabel("SFS bin (derived allele count)", fontsize=12)
    ax.set_ylabel("Proportion of segregating sites", fontsize=12)
    ax.set_title("SFS confounding: alpha vs. recent demography (Ne1-Ne7, 10-1000 gen)", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("sfs_confounding.png", dpi=150)
    plt.show()
    print("Saved sfs_confounding.png")


if __name__ == "__main__":

    # Run new simulations in parallel
    num_new_simulations = 100000
    num_workers = os.cpu_count() - 1
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
    import shap

    sfs_bin1_idx = 0   # SFS bin 1 (singletons) is x column 0
    # Ne1-Ne7 are theta columns 2-8; summarise recent demography as their mean
    ne_recent_vals = theta_np[:, 2:9].mean(axis=1)

    background_size = min(500, x_np.shape[0])
    rng_shap = np.random.default_rng(0)
    bg_idx = rng_shap.choice(x_np.shape[0], size=background_size, replace=False)
    X_bg = x_np[bg_idx]
    ne_recent_bg = ne_recent_vals[bg_idx]
    singleton_bg = x_np[bg_idx, sfs_bin1_idx]

    ncols = 3
    nrows = math.ceil(len(param_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for pi, name in enumerate(param_names):
        ax = axes[pi // ncols][pi % ncols]
        explainer = shap.TreeExplainer(rf_models[name], X_bg)
        shap_vals = explainer.shap_values(X_bg)   # (background_size, n_stats)
        shap_for_bin1 = shap_vals[:, sfs_bin1_idx]

        sc = ax.scatter(ne_recent_bg, shap_for_bin1,
                        c=singleton_bg, cmap="RdBu_r",
                        s=12, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("SFS bin 1 value", fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean Ne1-Ne7 (recent Ne multiplier)", fontsize=9)
        ax.set_ylabel(f"SHAP value of SFS bin 1\nfor RF({name})", fontsize=8)
        ax.set_title(f"Parameter: {name}", fontsize=10)

    for pi in range(len(param_names), nrows * ncols):
        axes[pi // ncols][pi % ncols].axis("off")

    plt.suptitle(
        "SHAP: how SFS singleton bin 1 drives each parameter's RF prediction\n"
        "as a function of recent Ne (mean Ne1-Ne7) — confounding diagnostic",
        y=1.01, fontsize=11
    )
    plt.tight_layout()
    plt.savefig("shap_alpha_Ne_interaction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved shap_alpha_Ne_interaction.png")

    # ----------------------------------------
    # POSTERIOR PREDICTIVE CHECK
    # ----------------------------------------
    def run_ppc_sim(params_np):
        try:
            p = torch.tensor(params_np, dtype=torch.float32)
            return simulator(p).numpy()
        except Exception:
            return None

    n_ppc = 500
    ppc_params = posterior.sample((n_ppc,), x=x_obs)
    ppc_params_list = [ppc_params[j].numpy() for j in range(n_ppc)]

    print(f"Running {n_ppc} PPC simulations across {num_workers} cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        ppc_results = list(tqdm(executor.map(run_ppc_sim, ppc_params_list),
                                total=n_ppc, desc="PPC simulations"))

    ppc_stats = np.array([r for r in ppc_results if r is not None])  # shape (n_valid, n_stats)
    x_obs_np = x_obs.numpy()

    # Plot first 20 summary statistics
    n_plot = min(20, n_stats)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for k, ax in enumerate(axes.flat):
        if k >= n_plot:
            ax.axis("off")
            continue
        ax.hist(ppc_stats[:, k], bins=30, density=True, color="steelblue", alpha=0.7, label="PPC")
        ax.axvline(x_obs_np[k], color="red", linewidth=2, label="Observed")
        ax.set_title(f"Stat {k}")
        if k == 0:
            ax.legend(fontsize=8)
    plt.suptitle("Posterior predictive check (first 20 summary statistics)", y=1.01)
    plt.tight_layout()
    plt.savefig("posterior_predictive_check.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved posterior_predictive_check.png")

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