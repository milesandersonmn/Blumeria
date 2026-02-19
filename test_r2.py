import msprime
import numpy as np
import math
import tskit
import scipy
import allel
from pandas import DataFrame
import sys
sys.path.append('/Users/milesanderson/PhD/Blumeria/')
import helper_functions

# -------------------
# Ne Parameters per Alpha value
# -------------------

Ne_1_9 = 40000
Ne_1_7 = 50000
Ne_1_5 = 60000
Ne_1_3 = 100000
Ne_1_1 = 4220562531

# -------------------
# Global parameters
# -------------------

sample_size = 43
target_min = 2860
target_max = 2900
mu = 5e-7
r_chrom = 3.37e-7
file = "summary_statistics_growth_mu5e-7_rho3e-7_2860_2900_unfoldedSFS.csv"
growth_rate_low = 0.000001
growth_rate_high = 0.0001
exclude_ac_below = 13
ploidy = 1
#growth_rate_low = 0
#growth_rate_high = 0

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


def alpha1_9(arg):
    
    global Ne_1_9

    alpha = 1.9
    Ne = Ne_1_9

    growth_rate = np.random.uniform(low = growth_rate_low, high = growth_rate_high)
    print("growth rate:", growth_rate)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    rng = np.random.default_rng(12345)
    attempt = 0
    while True:
        attempt += 1
        ts = helper_functions.instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map, sample_size = sample_size)
        mts = helper_functions.mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne_1_9 = Ne
            np.set_printoptions(legacy="1.21") #exclude dbtype from np arrays
            summary_statistics = [] #Initialize list of summary statistics

            if alpha == 1.9:
                summary_statistics.append(1) #First column corresponds to model index
            elif alpha == 1.7:
                summary_statistics.append(2)
            elif alpha == 1.5:
                summary_statistics.append(3)
            elif alpha == 1.3:
                summary_statistics.append(4)
            elif alpha == 1.1:
                summary_statistics.append(5)

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
            print("Summary stat length = ",len(summary_statistics))
            print(summary_statistics)

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
                print(chrom_ts.sequence_length)



            s, mask = helper_functions.calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below, sample_size = sample_size)
            
            print("calculated r2")

            t, maks = helper_functions.calculate_r2_vectorized(mts = mts, exclude_ac_below = exclude_ac_below, sample_size = sample_size)

            print(s == t)
            print("s:")
            print(s)
            print("t:")
            print(t)
            print(mask == maks)
            break

alpha1_9(1)
