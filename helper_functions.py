import msprime
import numpy as np
import math
import tskit
import scipy
import allel
from pandas import DataFrame

#--------------------
#Helper Functions
#--------------------

#######Instantiate a beta coalescent demography with msprime
def instantiate_ts(alpha, initial_size, growth_rate, rate_map, sample_size, ploidy = 1):
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

######Define function to calculate r2 between two SNP loci
def ld_r2(a, b):
    pa = a.mean()
    pb = b.mean()
    pab = np.mean((a == 1) & (b == 1))
    
    D = pab - pa * pb
    denom = pa * (1 - pa) * pb * (1 - pb)
    
    return 0 if denom == 0 else (D * D) / denom

######Calculate pairwise r-squared for entire genome with allele count pruning
"""
exclude_ac_below argument will prune all variants below that allele count
(i.e. allele count of 1 will exclude singletons from r-squared calculation, 2 = doubletons, etc.)
"""
def calculate_r2(mts, exclude_ac_below, sample_size):
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
    gn_filt = g[mask].astype(np.float32)
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

######Calculate r^2 using vecotrized operations

def calculate_r2_vectorized(mts, exclude_ac_below, sample_size):
    print("Converted to genotype matrix...")
    g = mts.genotype_matrix()
    g = allel.HaplotypeArray(g)

    # Count alternate allele occurrences
    ac = g.count_alleles(max_allele=1)

    # Filter mask
    mask = (ac[:, 1] > exclude_ac_below) & (ac[:, 1] <= sample_size - exclude_ac_below)

    gn_filt = g[mask].astype(np.float32)

    print("Shape of unfiltered genotype matrix =", g.shape)
    print("Shape of filtered matrix =", gn_filt.shape)

    # ---- Vectorized r^2 ----
    X = gn_filt  # shape (n_snps, n_haplotypes)

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
    s = np.zeros_like(D, dtype=np.float32)
    np.divide(D * D, denom, out=s, where=denom != 0)
    np.fill_diagonal(s, 0.0)
    return s, mask


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