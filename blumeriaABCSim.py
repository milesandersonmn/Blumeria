import msprime
import numpy as np
import math
import tskit
import scipy
import allel
import pandas as pd

import os
os.chdir("/Users/milesanderson/PhD/Blumeria/summary_stats/growth")

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
#file = "summary_statistics_growth_mu5e-7_rho3e-7_2860_2900_unfoldedSFS.csv"
file = (
    f"summary_statistics_contraction_"
    f"mu{mu:.0e}_rho{r_chrom:.2e}_"
    f"{target_min}_{target_max}_unfoldedSFS.csv"
)
#growth_rate_low = 0.000001
#growth_rate_high = 0.0001
exclude_ac_below = 2
ploidy = 1
#growth_rate_low = 0
#growth_rate_high = 0
growth_rate_high = -0.000001
growth_rate_low = -0.0001

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
#Helper Functions
#--------------------

#######Instantiate a beta coalescent demography with msprime
def instantiate_ts(alpha, initial_size, growth_rate, rate_map, ploidy = 1):
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size, growth_rate=growth_rate)
    ts = msprime.sim_ancestry(
        samples = sample_size,
        demography=demography,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        ploidy=ploidy
        #random_seed=1234
    )
    return ts

######Add mutations

def mutation_model(ts,mu):
    mts = msprime.sim_mutations(
        ts,
        rate=mu
        #random_seed=rng.integers(1, 2**31 - 1)
    )
    return mts

######Perform AFS calculations and append to summary stats list

def add_sfs_summary(afs, sample_size, summary_statistics):
    """
    Builds afs_entries from an AFS vector and appends
    SFS summary statistics (bins 1 through 14 and 15+) directly to 
    an existing summary_statistics list.

    Parameters
    ----------
    afs : array-like
        AFS array where afs[x] = number of mutations with count x.
    sample_size : int
        Number of haploid individuals.
    summary_statistics : list
        Pre-existing list to append the SFS summary statistics to.

    Returns
    -------
    afs_entries : np.ndarray
        Flattened allele-frequency entries.
    """

    # Build afs_entries
    afs_entries = []
    total_chromosomes = sample_size 

    for x in range(1, len(afs)):
        num_mutations = afs[x]
        freq = x / total_chromosomes
        afs_entries.extend([freq] * int(num_mutations))

    afs_entries = np.array(afs_entries)

    # Append SFS summary stats directly to the existing list
    total = afs.sum()

    # bins 1–14
    for i in range(1, 15):
        summary_statistics.append(afs[i] / total)

    # bin 15+
    summary_statistics.append(afs[15:].sum() / total)

    return afs_entries

"""
######Define function to calculate r2 between two SNP loci
def ld_r2(a, b):
    pa = a.mean()
    pb = b.mean()
    pab = np.mean((a == 1) & (b == 1))
    
    D = pab - pa * pb
    denom = pa * (1 - pa) * pb * (1 - pb)
    
    return 0 if denom == 0 else (D * D) / denom
"""

######Calculate pairwise r-squared for entire genome with allele count pruning
"""
exclude_ac_below argument will prune all variants below that allele count
(i.e. allele count of 1 will exclude singletons from r-squared calculation, 2 = doubletons, etc.)
"""
"""
def calculate_r2(mts, exclude_ac_below):
    print("Converted to genotype matrix...")
    g = mts.genotype_matrix()

    g = allel.HaplotypeArray(g)
    # Count alternate allele occurrences at each variant
    ac = g.count_alleles(max_allele = 1)
    #print("AC")
    #print(ac[:, 1])
    # Identify n-tons (alt allele count == exclude_ac_below)
    mask = (ac[:, 1] > exclude_ac_below) & (ac[:, 1] <= sample_size-exclude_ac_below)
    #mask = ac[:, 1] > exclude_ac_below
    #print(mask)
    # Filter out singletons
    gn_filt = g[mask]
    #print("gn_filt:")
    #print(gn_filt)
    #print("Shape of mask =", np.shape(mask))
    print("Shape of unfiltered genotype matrix = ", np.shape(g))
    print("Shape of filtered matrix = ", np.shape(gn_filt))
    # Compute r only on filtered data

    n = gn_filt.shape[0]
    print("n shape", n)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            s[i,j] = s[j,i] = ld_r2(gn_filt[i,:], gn_filt[j,:])
    
    return s, mask
"""
def calculate_r2(mts, exclude_ac_below):
    print("Converted to genotype matrix...")
    g = mts.genotype_matrix()
    g = allel.HaplotypeArray(g)

    # Count alternate allele occurrences

    """ac = g.count_alleles(max_allele=1)

    # Filter mask
    mask = (ac[:, 1] > exclude_ac_below) & (ac[:, 1] <= sample_size - exclude_ac_below)

    gn_filt = g[mask].astype(np.float64)"""

    # Count *all* alleles
    ac_all = g.count_alleles(max_allele=None)

    # Keep only sites where max allele index == 1 (i.e. only 0/1 present)
    biallelic_mask = ac_all.max_allele() == 1

    # Apply both filters
    mask = biallelic_mask & (ac_all[:, 1] > exclude_ac_below) & (ac_all[:, 1] <= sample_size - exclude_ac_below)

    gn_filt = g[mask].astype(np.float64)


    print("Shape of unfiltered genotype matrix =", g.shape)
    print("Shape of filtered matrix =", gn_filt.shape)

    # ---- Vectorized r^2 ----
    X = gn_filt  # shape (n_snps, n_haplotypes)
    if not np.all((X == 0) | (X == 1)):
        print("WARNING: X contains values other than 0/1!")
        print("Unique values in X:", np.unique(X))

    n, m = X.shape

    # p_a for each SNP
    p = X.mean(axis=1)                         # shape (n,)

    # p_ab for all SNP pairs
    pab = (X @ X.T) / m                        # shape (n,n)

    # D = p_ab - p_a p_b
    D = pab - np.outer(p, p)

    # Denominator
    denom = np.outer(p * (1 - p), p * (1 - p))

    # r^2 with safe divide
    s = np.zeros_like(D, dtype=np.float64)
    np.divide(D * D, denom, out=s)#, where=denom != 0)
    np.fill_diagonal(s, 0.0)

    # --- r^2 max (using same p!) ---
    pA = p[:, None]
    pB = p[None, :]

    Dmax = np.minimum(pA * (1 - pB), (1 - pA) * pB)
    absDmin = np.minimum(pA * pB, (1 - pA) * (1 - pB))

    D2max = np.maximum(Dmax**2, absDmin**2)
    denom_max = pA * (1 - pA) * pB * (1 - pB)

    """r2_max = np.zeros_like(D2max, dtype=np.float64)
    np.divide(D2max, denom_max, out=r2_max)#, where=denom_max != 0)
    np.fill_diagonal(r2_max, 0.0)

    pA = p[:, None]
    pB = p[None, :]

    Dmax = np.minimum(pA * (1 - pB), (1 - pA) * pB)
    Dmin_abs = np.minimum(pA * pB, (1 - pA) * (1 - pB))

    D2max = np.minimum(Dmax**2, Dmin_abs**2)
    denom_max = pA * (1 - pA) * pB * (1 - pB)

    r2_max = np.zeros_like(D2max, dtype=np.float64)
    np.divide(D2max, denom_max, out=r2_max, where=denom_max > 1e-12)
    np.fill_diagonal(r2_max, 0.0)

    # --- normalized r^2 ---
    r2_norm = np.zeros_like(s, dtype=np.float64)
    np.divide(s, r2_max, out=r2_norm, where=r2_max > 0)
    np.fill_diagonal(r2_norm, 0.0)"""

    
    # r^2
    s = np.zeros_like(D, dtype=np.float64)
    np.divide(D * D, denom, out=s, where=denom > 1e-12)
    np.fill_diagonal(s, 0.0)

    # r^2 max
    r2_max = np.zeros_like(D2max, dtype=np.float64)
    np.divide(D2max, denom_max, out=r2_max, where=denom_max > 1e-12)
    np.fill_diagonal(r2_max, 0.0)

    # normalized r^2
    r2_norm = np.zeros_like(s, dtype=np.float64)
    np.divide(s, r2_max, out=r2_norm, where=r2_max > 1e-12)
    np.fill_diagonal(r2_norm, 0.0)

    # hard cap
    #s = np.minimum(s, 1.0)
    #r2_norm = np.minimum(r2_norm, 1.0)
    # ignore NaNs
    idx = np.nanargmax(r2_norm)

    # convert flat index to matrix indices
    i, j = np.unravel_index(idx, r2_norm.shape)

    print("Max r2_norm at indices:", i, j)
    print("r2_norm =", r2_norm[i, j])

    print("r2 =", s[i, j])
    print("r2_max =", r2_max[i, j])

    print("p_i =", p[i])
    print("p_j =", p[j])


    return s, r2_norm, mask


######Calculate hamming distance stats
def calculate_hamming(h, ploidy, summary_statistics):
    h = h.to_genotypes(ploidy=ploidy)
    h = h.to_n_alt(fill=-1)
    
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #38 hamming quantile 0.1
    summary_statistics.append(hamming_quant[1]) #39 0.3
    summary_statistics.append(hamming_quant[2]) #40 0.5
    summary_statistics.append(hamming_quant[3]) #41 0.7
    summary_statistics.append(hamming_quant[4]) #42 0.9
    summary_statistics.append(np.nanmean(hamming_array)) #43 mean hamming
    summary_statistics.append(np.nanstd(hamming_array)) #44 std hamming
    summary_statistics.append(np.nanvar(hamming_array)) #45 var hamming
    print("Sum_stats with hamming")
    print(summary_statistics)

######Calculate length of homozygous regions

def calculate_homozygous_lengths(ts_chroms, arr):
        #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = arr[:chrom1_mut_num]

    differences = [chrom1_homozygous[i+1] - chrom1_homozygous[i] for i in range(len(chrom1_homozygous)-1)]

    #print(differences)

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = arr[chrom1_mut_num:chrom1and2_mut_num]

    differences = [chrom2_homozygous[i+1] - chrom2_homozygous[i] for i in range(len(chrom2_homozygous)-1)]

    #print(differences)

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = arr[chrom1and2_mut_num:total_mut_num]

    differences = [chrom3_homozygous[i+1] - chrom3_homozygous[i] for i in range(len(chrom3_homozygous)-1)]
    return differences

######Get r-squared on all chromosomes independently with singletons masked

def get_rsq_per_chromosome(mask, ts_chroms, s):

    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)
    print("Mutation numbers:")
    print(chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, len(s))
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]


    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))

    return r2

######Get Rsq_norm

def get_rsq_norm_per_chromosome(mask, ts_chroms, r2_norm):

    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)
    print("Mutation numbers:")
    print(chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, len(r2_norm))
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    chrom1_ld = r2_norm[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_ld = r2_norm[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_ld = r2_norm[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]


    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2_norm = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    print("max R2 norm:")
    print(max(r2_norm))
    return r2_norm

######Get ILD

def get_ILD(mask, ts_chroms, s):
    
    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    print("ILD masks lens:")
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    # Chromosome index boundaries
    c1 = chrom1_mut_num
    c2 = chrom1and2_mut_num
    c3 = total_mut_num   # total mutation count
    print("index boundaries:")
    print(c1,c2,c3)
    # Block slices:
    # Chrom1 = [0:c1]
    # Chrom2 = [c1:c2]
    # Chrom3 = [c2:c3]

    # ---- Only take each inter-chromosome comparison once ---- #
    print("s")
    print(s)
    print("shape of s:", np.array(s).shape)
    # Chrom1 vs Chrom2  (upper-right block)
    chrom1_2 = s[0:c1, c1:c2].ravel()
    print(chrom1_2)
    # Chrom1 vs Chrom3
    chrom1_3 = s[0:c1, c2:c3].ravel()

    # Chrom2 vs Chrom3
    chrom2_3 = s[c1:c2, c2:c3].ravel()

    ild_all = np.concatenate((chrom1_2, chrom1_3, chrom2_3))

    return ild_all

######Normalized ILD

def get_ILD_norm(mask, ts_chroms, r2_norm):
    
    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    print("ILD masks lens:")
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    # Chromosome index boundaries
    c1 = chrom1_mut_num
    c2 = chrom1and2_mut_num
    c3 = total_mut_num   # total mutation count
    print("index boundaries:")
    print(c1,c2,c3)
    # Block slices:
    # Chrom1 = [0:c1]
    # Chrom2 = [c1:c2]
    # Chrom3 = [c2:c3]

    # ---- Only take each inter-chromosome comparison once ---- #
    print("r2_norm")
    print(r2_norm)
    print("shape of r2_norm:", np.array(r2_norm).shape)
    # Chrom1 vs Chrom2  (upper-right block)
    chrom1_2 = r2_norm[0:c1, c1:c2].ravel()
    print(chrom1_2)
    # Chrom1 vs Chrom3
    chrom1_3 = r2_norm[0:c1, c2:c3].ravel()

    # Chrom2 vs Chrom3
    chrom2_3 = r2_norm[c1:c2, c2:c3].ravel()

    ild_norm_all = np.concatenate((chrom1_2, chrom1_3, chrom2_3))

    return ild_norm_all

######Calculate Anderson's r-squared

def calculate_Anderson_rsq(mask, ts_chroms, s):

    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    print(chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, len(s))
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]


    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]

    arr_chrom1 = ts_chroms[0].sites_position[mask_chrom1] #array of site positions
    pairwise_distances_chrom1 = abs(arr_chrom1[:, None] - arr_chrom1) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom1 = pairwise_distances_chrom1[np.triu_indices_from(pairwise_distances_chrom1, k=1)]

    chrom1_scaled_ld = np.multiply(chrom1_ld, pairwise_distances_chrom1)

    ################Chrom2

    arr_chrom2 = ts_chroms[1].sites_position[mask_chrom2] #array of site positions
    pairwise_distances_chrom2 = abs(arr_chrom2[:, None] - arr_chrom2) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom2 = pairwise_distances_chrom2[np.triu_indices_from(pairwise_distances_chrom2, k=1)]


    chrom2_scaled_ld = np.multiply(chrom2_ld, pairwise_distances_chrom2)

    #############Chrom3

    arr_chrom3 = ts_chroms[2].sites_position[mask_chrom3] #array of site positions
    pairwise_distances_chrom3 = abs(arr_chrom3[:, None] - arr_chrom3) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom3 = pairwise_distances_chrom3[np.triu_indices_from(pairwise_distances_chrom3, k=1)]


    chrom3_scaled_ld = np.multiply(chrom3_ld, pairwise_distances_chrom3)
    

    if len(mask_chrom1[mask_chrom1 != False]) != len(arr_chrom1) or len(mask_chrom2[mask_chrom2 != False]) != len(arr_chrom2) or len(mask_chrom3[mask_chrom3 != False]) != len(arr_chrom3):
        raise ValueError("Mask length does not equal pairwise distance array length")

    scaled_r2 = np.concatenate((chrom1_scaled_ld, chrom2_scaled_ld, chrom3_scaled_ld))

    return scaled_r2

###### Normalized Anderson's R2

def calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm):

    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    print(chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, len(r2_norm))
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    chrom1_ld = r2_norm[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_ld = r2_norm[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_ld = r2_norm[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]


    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]

    arr_chrom1 = ts_chroms[0].sites_position[mask_chrom1] #array of site positions
    pairwise_distances_chrom1 = abs(arr_chrom1[:, None] - arr_chrom1) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom1 = pairwise_distances_chrom1[np.triu_indices_from(pairwise_distances_chrom1, k=1)]

    chrom1_scaled_ld = np.multiply(chrom1_ld, pairwise_distances_chrom1)

    ################Chrom2

    arr_chrom2 = ts_chroms[1].sites_position[mask_chrom2] #array of site positions
    pairwise_distances_chrom2 = abs(arr_chrom2[:, None] - arr_chrom2) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom2 = pairwise_distances_chrom2[np.triu_indices_from(pairwise_distances_chrom2, k=1)]


    chrom2_scaled_ld = np.multiply(chrom2_ld, pairwise_distances_chrom2)

    #############Chrom3

    arr_chrom3 = ts_chroms[2].sites_position[mask_chrom3] #array of site positions
    pairwise_distances_chrom3 = abs(arr_chrom3[:, None] - arr_chrom3) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom3 = pairwise_distances_chrom3[np.triu_indices_from(pairwise_distances_chrom3, k=1)]


    chrom3_scaled_ld = np.multiply(chrom3_ld, pairwise_distances_chrom3)
    

    if len(mask_chrom1[mask_chrom1 != False]) != len(arr_chrom1) or len(mask_chrom2[mask_chrom2 != False]) != len(arr_chrom2) or len(mask_chrom3[mask_chrom3 != False]) != len(arr_chrom3):
        raise ValueError("Mask length does not equal pairwise distance array length")

    scaled_r2_norm = np.concatenate((chrom1_scaled_ld, chrom2_scaled_ld, chrom3_scaled_ld))

    return scaled_r2_norm


#--------------------
#Alpha Functions
#--------------------
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
        ts = instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map)
        mts = mutation_model(ts = ts, mu = mu)
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

            afs_entries = add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
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



            s, s_norm, mask = calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below)
            
            print("calculated r2")

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            print("Haplotypes...", h)
            print("to genotypes...", h.to_genotypes(ploidy=1))

            """
            summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #35 mean inbreeding
            summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #36 std inbreeding
            summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #37 var inbreeding
            """
            print("Calculating hamming distance...")
            calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
            print(len(summary_statistics))
            
            differences = calculate_homozygous_lengths(ts_chroms = ts_chroms, arr = arr)

            homozygous_quant = np.nanquantile(differences, [0.1,0.3,0.5,0.7,0.9])


            summary_statistics.append(homozygous_quant[0]) #46-50 lengths of homozygosity quantiles
            summary_statistics.append(homozygous_quant[1])
            summary_statistics.append(homozygous_quant[2])
            summary_statistics.append(homozygous_quant[3])
            summary_statistics.append(homozygous_quant[4])
            summary_statistics.append(np.nanmean(differences)) #51 mean homozygosity length
            #summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))   
            summary_statistics.append(np.nanvar(differences)) #52 var homozygosity
            summary_statistics.append(np.nanstd(differences)) #53 std homozygosity


            r2 = get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            print("Fraction of r2 values >= 1:", r2_ge_1)
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
            print(r2)
            print(len(r2))
            print(len(r2[r2 != 0]))
            print(len(summary_statistics))


            ild_all = get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            print("Fraction of ILD values >= 1:", ild_ge_1)
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
            print(ild_all)
            print("ILD")
            

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
            print(len(summary_statistics))

            print(ild_all)
            print(len(ild_all))
            print(len(ild_all[ild_all != 0]))
            

            scaled_r2 = calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
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
            print(len(summary_statistics))


            r2_norm = get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
            print("Fraction of r2_norm values >= 1:", r2_norm_ge_1)
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
            print(len(summary_statistics))     

            ild_norm_all = get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
            print("Fraction of ild_norm values >= 1:", ild_norm_ge_1)     
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
            print(len(summary_statistics)) 

            scaled_r2_norm = calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
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

            print(result)
            print(np.mean(result))
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))

            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
            print(proportions)

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
            print(proportions) 
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies


            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            print(proportions) 
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            print(proportions) 
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
            print(summary_statistics)


            return summary_statistics
            #return pd.DataFrame(summary_statistics).T
            #x = DataFrame(summary_statistics).T
            #print(x)
            #x.to_csv(file, index = False, mode = 'a', header = False)
            

            break


def alpha1_7(arg):
    global Ne_1_7
    
    alpha = 1.7
    Ne = Ne_1_7

    growth_rate = np.random.uniform(low = growth_rate_low, high = growth_rate_high)
    print("growth rate:", growth_rate)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    rng = np.random.default_rng(12345)
    attempt = 0
    while True:
        attempt += 1
        ts = instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map)
        mts = mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne_1_7 = Ne
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

            afs_entries = add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
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



            s, s_norm, mask = calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below)
            
            print("calculated r2")

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            print("Haplotypes...", h)
            print("to genotypes...", h.to_genotypes(ploidy=1))

            """
            summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #35 mean inbreeding
            summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #36 std inbreeding
            summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #37 var inbreeding
            """
            print("Calculating hamming distance...")
            calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
            print(len(summary_statistics))
            
            differences = calculate_homozygous_lengths(ts_chroms = ts_chroms, arr = arr)

            homozygous_quant = np.nanquantile(differences, [0.1,0.3,0.5,0.7,0.9])


            summary_statistics.append(homozygous_quant[0]) #46-50 lengths of homozygosity quantiles
            summary_statistics.append(homozygous_quant[1])
            summary_statistics.append(homozygous_quant[2])
            summary_statistics.append(homozygous_quant[3])
            summary_statistics.append(homozygous_quant[4])
            summary_statistics.append(np.nanmean(differences)) #51 mean homozygosity length
            #summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))   
            summary_statistics.append(np.nanvar(differences)) #52 var homozygosity
            summary_statistics.append(np.nanstd(differences)) #53 std homozygosity


            r2 = get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            print("Fraction of r2 values >= 1:", r2_ge_1)
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
            print(r2)
            print(len(r2))
            print(len(r2[r2 != 0]))
            print(len(summary_statistics))


            ild_all = get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            print("Fraction of ILD values >= 1:", ild_ge_1)
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
            print(ild_all)
            print("ILD")
            

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
            print(len(summary_statistics))

            print(ild_all)
            print(len(ild_all))
            print(len(ild_all[ild_all != 0]))
            

            scaled_r2 = calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
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
            print(len(summary_statistics))


            r2_norm = get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
            print("Fraction of r2_norm values >= 1:", r2_norm_ge_1)
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
            print(len(summary_statistics))     

            ild_norm_all = get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
            print("Fraction of ild_norm values >= 1:", ild_norm_ge_1)     
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
            print(len(summary_statistics)) 

            scaled_r2_norm = calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
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

            print(result)
            print(np.mean(result))
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))

            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
            print(proportions)

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
            print(proportions) 
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies
            
            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            print(proportions) 
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            print(proportions) 
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
            print(summary_statistics)                        

            return summary_statistics
            #return pd.DataFrame(summary_statistics).T
            #x = DataFrame(summary_statistics).T
            #print(x)
            #x.to_csv(file, index = False, mode = 'a', header = False)
            

            break



def alpha1_5(arg):
    global Ne_1_5
    
    alpha = 1.5
    
    Ne = Ne_1_5
    #mu = 1e-9                    # mutation rate per bp per generation
    #target_min, target_max = 1100, 1140
    random_seed = 42

    #r_chrom = 1e-9 #Recombination rate
    r_break = math.log(2) #Recombination rate needed to satisfy probability 2^-t inheritance of two chromsomes
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

    growth_rate = np.random.uniform(low = growth_rate_low, high = growth_rate_high)
    print("growth rate:", growth_rate)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    rng = np.random.default_rng(12345)
    attempt = 0
    while True:
        attempt += 1
        ts = instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map)
        mts = mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne_1_5 = Ne
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

            afs_entries = add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
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



            s, s_norm, mask = calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below)
            
            print("calculated r2")

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            print("Haplotypes...", h)
            print("to genotypes...", h.to_genotypes(ploidy=1))

            """
            summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #35 mean inbreeding
            summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #36 std inbreeding
            summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #37 var inbreeding
            """
            print("Calculating hamming distance...")
            calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
            print(len(summary_statistics))
            
            differences = calculate_homozygous_lengths(ts_chroms = ts_chroms, arr = arr)

            homozygous_quant = np.nanquantile(differences, [0.1,0.3,0.5,0.7,0.9])


            summary_statistics.append(homozygous_quant[0]) #46-50 lengths of homozygosity quantiles
            summary_statistics.append(homozygous_quant[1])
            summary_statistics.append(homozygous_quant[2])
            summary_statistics.append(homozygous_quant[3])
            summary_statistics.append(homozygous_quant[4])
            summary_statistics.append(np.nanmean(differences)) #51 mean homozygosity length
            #summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))   
            summary_statistics.append(np.nanvar(differences)) #52 var homozygosity
            summary_statistics.append(np.nanstd(differences)) #53 std homozygosity


            r2 = get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            print("Fraction of r2 values >= 1:", r2_ge_1)
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
            print(r2)
            print(len(r2))
            print(len(r2[r2 != 0]))
            print(len(summary_statistics))


            ild_all = get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            print("Fraction of ILD values >= 1:", ild_ge_1)
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
            print(ild_all)
            print("ILD")
            

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
            print(len(summary_statistics))

            print(ild_all)
            print(len(ild_all))
            print(len(ild_all[ild_all != 0]))
            

            scaled_r2 = calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
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
            print(len(summary_statistics))


            r2_norm = get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
            print("Fraction of r2_norm values >= 1:", r2_norm_ge_1)
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
            print(len(summary_statistics))     

            ild_norm_all = get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
            print("Fraction of ild_norm values >= 1:", ild_norm_ge_1)     
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
            print(len(summary_statistics)) 

            scaled_r2_norm = calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
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

            print(result)
            print(np.mean(result))
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))

            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
            print(proportions)

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
            print(proportions) 
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies
                        
            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            print(proportions) 
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            print(proportions) 
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
            print(summary_statistics)
            return summary_statistics
            #return pd.DataFrame(summary_statistics).T
            #x = DataFrame(summary_statistics).T
            #print(x)
            #x.to_csv(file, index = False, mode = 'a', header = False)
            

            break



def alpha1_3(arg):
    global Ne_1_3
    
    alpha = 1.3
    
    Ne = Ne_1_3
    #mu = 1e-9                  # mutation rate per bp per generation
    #target_min, target_max = 1100, 1140
    random_seed = 42

    #r_chrom = 1e-9 #Recombination rate
    r_break = math.log(2) #Recombination rate needed to satisfy probability 2^-t inheritance of two chromsomes
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

    growth_rate = np.random.uniform(low = growth_rate_low, high = growth_rate_high)
    print("growth rate:", growth_rate)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    rng = np.random.default_rng(12345)
    attempt = 0
    while True:
        attempt += 1
        ts = instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map)
        mts = mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne_1_3 = Ne
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

            afs_entries = add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
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



            s, s_norm, mask = calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below)
            
            print("calculated r2")

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            print("Haplotypes...", h)
            print("to genotypes...", h.to_genotypes(ploidy=1))

            """
            summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #35 mean inbreeding
            summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #36 std inbreeding
            summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #37 var inbreeding
            """
            print("Calculating hamming distance...")
            calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
            print(len(summary_statistics))
            
            differences = calculate_homozygous_lengths(ts_chroms = ts_chroms, arr = arr)

            homozygous_quant = np.nanquantile(differences, [0.1,0.3,0.5,0.7,0.9])


            summary_statistics.append(homozygous_quant[0]) #46-50 lengths of homozygosity quantiles
            summary_statistics.append(homozygous_quant[1])
            summary_statistics.append(homozygous_quant[2])
            summary_statistics.append(homozygous_quant[3])
            summary_statistics.append(homozygous_quant[4])
            summary_statistics.append(np.nanmean(differences)) #51 mean homozygosity length
            #summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))   
            summary_statistics.append(np.nanvar(differences)) #52 var homozygosity
            summary_statistics.append(np.nanstd(differences)) #53 std homozygosity


            r2 = get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            print("Fraction of r2 values >= 1:", r2_ge_1)
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
            print(r2)
            print(len(r2))
            print(len(r2[r2 != 0]))
            print(len(summary_statistics))


            ild_all = get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            print("Fraction of ILD values >= 1:", ild_ge_1)
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
            print(ild_all)
            print("ILD")
            

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
            print(len(summary_statistics))

            print(ild_all)
            print(len(ild_all))
            print(len(ild_all[ild_all != 0]))
            

            scaled_r2 = calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
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
            print(len(summary_statistics))


            r2_norm = get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
            print("Fraction of r2_norm values >= 1:", r2_norm_ge_1)
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
            print(len(summary_statistics))     

            ild_norm_all = get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
            print("Fraction of ild_norm values >= 1:", ild_norm_ge_1)     
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
            print(len(summary_statistics)) 

            scaled_r2_norm = calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
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

            print(result)
            print(np.mean(result))
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))

            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
            print(proportions)

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
            print(proportions) 
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies
                        
            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            print(proportions) 
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            print(proportions) 
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
            print(summary_statistics)
            return summary_statistics
            #return pd.DataFrame(summary_statistics).T
            #x = DataFrame(summary_statistics).T
            #print(x)
            #x.to_csv(file, index = False, mode = 'a', header = False)
            

            break



def alpha1_1(arg):
    global Ne_1_1
    
    alpha = 1.1
   
    Ne = Ne_1_1
    #mu = 1e-9                    # mutation rate per bp per generation
    #target_min, target_max = 1100, 1140
    random_seed = 42

    #r_chrom = 1e-9 #Recombination rate
    r_break = math.log(2) #Recombination rate needed to satisfy probability 2^-t inheritance of two chromsomes
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

    growth_rate = np.random.uniform(low = growth_rate_low, high = growth_rate_high)
    print("growth rate:", growth_rate)
    # -------------------
    # REJECTION SAMPLING OVER MUTATIONS
    # -------------------
    rng = np.random.default_rng(12345)
    attempt = 0
    while True:
        attempt += 1
        ts = instantiate_ts(alpha = alpha, initial_size = Ne, growth_rate = growth_rate, rate_map = rate_map)
        mts = mutation_model(ts = ts, mu = mu)
        S = mts.num_sites
        
        if target_min > S:
            Ne =(target_min/S)*Ne
        if target_max < S:
            Ne = (target_max/S)*Ne
        if target_min <= S <= target_max:
            print(f"Accepted on attempt {attempt}: S = {S}; Ne = {Ne}")
            Ne_1_1 = Ne
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

            afs_entries = add_sfs_summary(afs = afs, sample_size = sample_size, summary_statistics = summary_statistics) #12:26 SFS
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



            s, s_norm, mask = calculate_r2(mts = mts, exclude_ac_below = exclude_ac_below)
            
            print("calculated r2")

            arr = mts.sites_position #array of site positions

            h = allel.HaplotypeArray(mts.genotype_matrix())
            print("Haplotypes...", h)
            print("to genotypes...", h.to_genotypes(ploidy=1))

            """
            summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #35 mean inbreeding
            summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #36 std inbreeding
            summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=1)))) #37 var inbreeding
            """
            print("Calculating hamming distance...")
            calculate_hamming(h = h, ploidy = 1, summary_statistics = summary_statistics) #38-45 Hamming stats
            print(len(summary_statistics))
            
            differences = calculate_homozygous_lengths(ts_chroms = ts_chroms, arr = arr)

            homozygous_quant = np.nanquantile(differences, [0.1,0.3,0.5,0.7,0.9])


            summary_statistics.append(homozygous_quant[0]) #46-50 lengths of homozygosity quantiles
            summary_statistics.append(homozygous_quant[1])
            summary_statistics.append(homozygous_quant[2])
            summary_statistics.append(homozygous_quant[3])
            summary_statistics.append(homozygous_quant[4])
            summary_statistics.append(np.nanmean(differences)) #51 mean homozygosity length
            #summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))   
            summary_statistics.append(np.nanvar(differences)) #52 var homozygosity
            summary_statistics.append(np.nanstd(differences)) #53 std homozygosity


            r2 = get_rsq_per_chromosome(mask = mask, ts_chroms = ts_chroms, s = s)
            r2_ge_1 = np.mean(r2 >= 1)
            print("Fraction of r2 values >= 1:", r2_ge_1)
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
            print(r2)
            print(len(r2))
            print(len(r2[r2 != 0]))
            print(len(summary_statistics))


            ild_all = get_ILD(mask = mask, ts_chroms = ts_chroms, s = s)
            ild_ge_1 = np.mean(ild_all >= 1)
            print("Fraction of ILD values >= 1:", ild_ge_1)
            ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9,0.95,0.99])
            print(ild_all)
            print("ILD")
            

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
            print(len(summary_statistics))

            print(ild_all)
            print(len(ild_all))
            print(len(ild_all[ild_all != 0]))
            

            scaled_r2 = calculate_Anderson_rsq(mask = mask, ts_chroms = ts_chroms, s = s)
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
            print(len(summary_statistics))


            r2_norm = get_rsq_norm_per_chromosome(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)
            r2_norm_ge_1 = np.mean(r2_norm >= 1)
            print("Fraction of r2_norm values >= 1:", r2_norm_ge_1)
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
            print(len(summary_statistics))     

            ild_norm_all = get_ILD_norm(mask = mask, ts_chroms = ts_chroms, r2_norm = s_norm)  
            ild_norm_ge_1 = np.mean(ild_norm_all >= 1)
            print("Fraction of ild_norm values >= 1:", ild_norm_ge_1)     
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
            print(len(summary_statistics)) 

            scaled_r2_norm = calculate_Anderson_rsq_norm(mask, ts_chroms, r2_norm = s_norm) 
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

            print(result)
            print(np.mean(result))
            summary_statistics.append(np.nanmean(result)) #102-103 normalized Tajima's D
            summary_statistics.append(np.nanstd(result))

            #Clip r^2 values greater than 1
            r2 = np.clip(r2, 0, 1)
            proportions = np.histogram(r2, bins=np.arange(0, 1.1, 0.1))[0] / len(r2)
            print(proportions)

            summary_statistics.extend(proportions) #104-113 LD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #114 is difference between unlinked and fully linked frequencies
            

            ild_all = np.clip(ild_all, 0, 1)

            proportions = np.histogram(ild_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_all)
            print(proportions) 
            summary_statistics.extend(proportions) #115-124 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #125 is difference between unlinked and fully linked frequencies

            r2_norm = np.clip(r2_norm, 0, 1) 

            proportions = np.histogram(r2_norm, bins=np.arange(0, 1.1, 0.1))[0] / len(r2_norm)
            print(proportions) 
            summary_statistics.extend(proportions) #126-135 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #136 is difference between unlinked and fully linked frequencies

            ild_norm_all = np.clip(ild_norm_all, 0, 1)
            proportions = np.histogram(ild_norm_all, bins=np.arange(0, 1.1, 0.1))[0] / len(ild_norm_all)
            print(proportions) 
            summary_statistics.extend(proportions) #137-146 ILD Frequency spectrum
            
            summary_statistics.append(proportions[0]-proportions[9]) #147 is difference between unlinked and fully linked frequencies
            print(summary_statistics)                        

            return summary_statistics
            #return pd.DataFrame(summary_statistics).T
            #x = DataFrame(summary_statistics).T
            #print(x)
            #x.to_csv(file, index = False, mode = 'a', header = False)
            

            break



import concurrent.futures
"""worker_num = 8
reps = 3200
with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_9, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_7, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_5, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_3, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_1, range(reps)))"""

"""worker_num = 8
reps = 1000  # or 3200 if testing

functions = [alpha1_9, alpha1_7, alpha1_5, alpha1_3, alpha1_1]

futures = []

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    
    # Submit ALL tasks (interleaved)
    for func in functions:
        for i in range(reps):
            futures.append(executor.submit(func, i))
    
    # Collect results as they complete
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

# Build final DataFrame once
final_df = pd.DataFrame(results)

# Write once
final_df.to_csv(file, index=False)"""

worker_num = 8
reps = 20000
functions = [alpha1_9, alpha1_7, alpha1_5, alpha1_3, alpha1_1]
#functions = [alpha1_5, alpha1_3, alpha1_1]
results_buffer = []
buffer_size = 5000   # write every 5k rows

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    futures = [executor.submit(func, i)
               for func in functions
               for i in range(reps)]

    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
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
