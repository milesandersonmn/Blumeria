import msprime
import numpy as np
import math
import scipy
import allel
import pandas as pd
import matplotlib.pyplot as plt
import helper_functions

import os
os.chdir("/Users/milesanderson/PhD/Blumeria/summary_stats/")


def simulate_beta_bottleneck(alpha, t_bottleneck, severity, duration,
                              N0,
                              n_samples, rate_map, ploidy = 1):
    print("Simming demography...")
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N0)
    
    # Add bottleneck as instantaneous size change
    demography.add_population_parameters_change(
        time=t_bottleneck, 
        population="A", 
        initial_size=N0 * severity  # severity << 1 for strong bottleneck
    )
    demography.add_population_parameters_change(
        time=t_bottleneck + duration,
        population="A",
        initial_size=N0  # recover
    )
    
    ts = msprime.sim_ancestry(
        samples=n_samples,
        demography=demography,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        ploidy=ploidy  # alpha in (1, 2)
    )
    return ts

# -------------------
# Ne Parameters per Alpha value
# -------------------



# -------------------
# Global parameters
# -------------------

sample_size = 43
target_min = 2860
target_max = 2900
mu = 5e-7
r_chrom = 3.37e-7
#file = "summary_statistics_growth_mu5e-7_rho3e-7_2860_2900_unfoldedSFS.csv"
"""
file = (
    f"summary_statistics_"
    f"mu{mu:.0e}_rho{r_chrom:.2e}_"
    f"{target_min}_{target_max}_unfoldedSFS.csv"
)
"""
file = "beta_bottleneck.csv"

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
rates = [r_chrom, r_break, r_chrom, r_break, r_chrom] 
rate_map = msprime.RateMap(position=map_positions, rate=rates) #Rate map for separate chromosomes


#--------------------
#Simulation function
#--------------------
def sim_summary_stats():
    print("Function called")
    t_b = np.random.uniform(10, 100000)
    print("t_b=", t_b)
    severity = np.random.uniform(0.001, 1)
    print("severity=", severity)
    duration = np.random.uniform(10, 5000)
    print("duration=", duration)

    
    Ne = 40000
    alpha = np.random.uniform(1.3, 1.8)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    
    attempt = 0
    while True:
        attempt += 1
        
        ts = simulate_beta_bottleneck(alpha=alpha, t_bottleneck=t_b, severity=severity, duration=duration, N0=Ne,
                                         n_samples=sample_size, rate_map=rate_map, )
        print("sim complete, simming mutations...")
        

        mts = helper_functions.mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne = round((target_min / S) * Ne)
            print(f"Ne too large")
        if target_max < S:
            Ne = round((target_max / S) * Ne)
            print(f"Ne too small")
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne = Ne
            np.set_printoptions(legacy="1.21") #exclude dbtype from np arrays
            summary_statistics = [] #Initialize list of summary statistics

            summary_statistics.append(3)

            summary_statistics.append(0) #Growth rate

            summary_statistics.append(Ne) #Second column is Ne

            summary_statistics.append(alpha) #Third column is alpha parameter

            summary_statistics.append(r_chrom/mu) #Fourth column is rho/theta

            S = mts.num_sites

            summary_statistics.append(S) #Fifth column is number of segregating sites
            normalized_S = mts.segregating_sites(span_normalise=True)
            summary_statistics.append(normalized_S) #Sixth column is span normalized S

            num_windows = 30
            pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
            summary_statistics.append(np.nanmean(pi_array)) #7 mean pi
            summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #8 harmonic mean pi
            summary_statistics.append(np.nanvar(pi_array)) #9 variance of pi
            summary_statistics.append(np.nanstd(pi_array)) #10 std pi
            pi = mts.diversity()
            summary_statistics.append(pi) #11 nucleotide diversity pi


            afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=True)

            afs_entries = helper_functions.add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
            

            afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
            summary_statistics.append(afs_quant[0]) #27 AFS quantile 0.1
            summary_statistics.append(afs_quant[1]) #28 0.3
            summary_statistics.append(afs_quant[2]) #29 0.5
            summary_statistics.append(afs_quant[3]) #30 0.7
            summary_statistics.append(afs_quant[4]) #31 0.9


            D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
            summary_statistics.append(np.nanmean(D_array)) #32 mean Tajima's D
            summary_statistics.append(np.nanvar(D_array)) #33 variance of Tajima's D
            summary_statistics.append(np.nanstd(D_array)) #34 std D

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
            summary_statistics.append(ild_ge_1)
            

         
            

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
            summary_statistics.append(ild_norm_ge_1)
           

            scaled_r2_norm = helper_functions.calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
            scaled_r2_norm_quant = np.nanquantile(scaled_r2_norm, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])


            summary_statistics.append(scaled_r2_norm_quant[0]) #94-101 Normalized ILD r^2 quantiles, mean, variance, std
            summary_statistics.append(scaled_r2_norm_quant[1])
            summary_statistics.append(scaled_r2_norm_quant[2])
            summary_statistics.append(scaled_r2_norm_quant[3])
            summary_statistics.append(scaled_r2_norm_quant[4])
            summary_statistics.append(scaled_r2_norm_quant[5])
            summary_statistics.append(scaled_r2_norm_quant[6])
            summary_statistics.append(np.nanmean(scaled_r2_norm))
            summary_statistics.append(np.nanvar(scaled_r2_norm))
            summary_statistics.append(np.nanstd(scaled_r2_norm))               
            
            
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

                adjacent_r2_stats['unweighted_quantiles'][0.1],
                adjacent_r2_stats['unweighted_quantiles'][0.3],
                adjacent_r2_stats['unweighted_quantiles'][0.5],
                adjacent_r2_stats['unweighted_quantiles'][0.7],
                adjacent_r2_stats['unweighted_quantiles'][0.9],
                adjacent_r2_stats['unweighted_mean'],
                adjacent_r2_stats['unweighted_std']

            ])

            #R2 norm of adjacent sites
            adjacent_norm_r2_stats = helper_functions.get_weighted_rsq_stats_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s_norm)
 
            summary_statistics.extend([
                adjacent_norm_r2_stats['weighted_quantiles'][0.1],
                adjacent_norm_r2_stats['weighted_quantiles'][0.3],
                adjacent_norm_r2_stats['weighted_quantiles'][0.5],
                adjacent_norm_r2_stats['weighted_quantiles'][0.7],
                adjacent_norm_r2_stats['weighted_quantiles'][0.9],
                adjacent_norm_r2_stats['weighted_mean'],
                adjacent_norm_r2_stats['weighted_std'],

                adjacent_norm_r2_stats['unweighted_quantiles'][0.1],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.3],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.5],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.7],
                adjacent_norm_r2_stats['unweighted_quantiles'][0.9],
                adjacent_norm_r2_stats['unweighted_mean'],
                adjacent_norm_r2_stats['unweighted_std']
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
           
            

            
#################
#Parallel calling
#################
        
import concurrent.futures
worker_num = 8
reps = 30000

functions = [sim_summary_stats]
results_buffer = []
buffer_size = 50   # write every 5k rows

print("Warming up...")
warmup_result = sim_summary_stats()
if warmup_result is not None:
    results_buffer.append(warmup_result)
print("Warmup complete, launching workers...")

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    futures = [executor.submit(func)
               for func in functions
               for _ in range(reps)]

    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                results_buffer.append(result)

            # Flush periodically
            if len(results_buffer) >= buffer_size:
                df = pd.DataFrame(results_buffer)
                df.to_csv(file, mode='a', index=False, header=not os.path.exists(file))
                results_buffer.clear()

        except Exception as e:
            print("Worker failed:", e)

# Flush any remaining results
if results_buffer:
    df = pd.DataFrame(results_buffer)
    df.to_csv(file, mode='a', index=False, header=not os.path.exists(file))
