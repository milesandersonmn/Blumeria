import allel
import os
import numpy as np
import scipy
os.chdir("/Users/milesanderson/PhD/Blumeria/")

exclude_ac_below = 2
ploidy = 1
sample_size = 43

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

def calculate_r2(g, exclude_ac_below):
    print("Converted to genotype matrix...")
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

def get_rsq_per_chromosome(mask, chrom, s):
    unique_chrom, counts = np.unique(chrom, return_counts=True)

    for c, n in zip(unique_chrom, counts):
        print(f"{c}: {n} sites")

    print(counts)
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)
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

def get_rsq_norm_per_chromosome(mask, chrom, r2_norm):

    unique_chrom, counts = np.unique(chrom, return_counts=True)

    for c, n in zip(unique_chrom, counts):
        print(f"{c}: {n} sites")

    print(counts)
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)
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

def get_ILD(mask, chrom, s):
    unique_chrom, counts = np.unique(chrom, return_counts=True)

    
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)

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

def get_ILD_norm(mask, chrom, r2_norm):

    unique_chrom, counts = np.unique(chrom, return_counts=True)

    for c, n in zip(unique_chrom, counts):
        print(f"{c}: {n} sites")

    print(counts)
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)

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

def calculate_Anderson_rsq(mask, pos, chrom, s):

    unique_chrom, counts = np.unique(chrom, return_counts=True)

    
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)

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
    
    arr_chrom1 = pos[chrom == unique_chrom[0]][mask_chrom1]
    print("arr1:", arr_chrom1)
    print(len(arr_chrom1))
    #arr_chrom1 = ts_chroms[0].sites_position[mask_chrom1] #array of site positions
    pairwise_distances_chrom1 = abs(arr_chrom1[:, None] - arr_chrom1) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom1 = pairwise_distances_chrom1[np.triu_indices_from(pairwise_distances_chrom1, k=1)]

    chrom1_scaled_ld = np.multiply(chrom1_ld, pairwise_distances_chrom1)

    ################Chrom2
    arr_chrom2 = pos[chrom == unique_chrom[1]][mask_chrom2]
    #arr_chrom2 = ts_chroms[1].sites_position[mask_chrom2] #array of site positions
    pairwise_distances_chrom2 = abs(arr_chrom2[:, None] - arr_chrom2) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom2 = pairwise_distances_chrom2[np.triu_indices_from(pairwise_distances_chrom2, k=1)]


    chrom2_scaled_ld = np.multiply(chrom2_ld, pairwise_distances_chrom2)

    #############Chrom3
    arr_chrom3 = pos[chrom == unique_chrom[2]][mask_chrom3]
    #arr_chrom3 = ts_chroms[2].sites_position[mask_chrom3] #array of site positions
    pairwise_distances_chrom3 = abs(arr_chrom3[:, None] - arr_chrom3) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom3 = pairwise_distances_chrom3[np.triu_indices_from(pairwise_distances_chrom3, k=1)]


    chrom3_scaled_ld = np.multiply(chrom3_ld, pairwise_distances_chrom3)
    

    if len(mask_chrom1[mask_chrom1 != False]) != len(arr_chrom1) or len(mask_chrom2[mask_chrom2 != False]) != len(arr_chrom2) or len(mask_chrom3[mask_chrom3 != False]) != len(arr_chrom3):
        raise ValueError("Mask length does not equal pairwise distance array length")

    scaled_r2 = np.concatenate((chrom1_scaled_ld, chrom2_scaled_ld, chrom3_scaled_ld))

    return scaled_r2

###### Normalized Anderson's R2

def calculate_Anderson_rsq_norm(mask, chrom, pos, r2_norm):

    unique_chrom, counts = np.unique(chrom, return_counts=True)

    for c, n in zip(unique_chrom, counts):
        print(f"{c}: {n} sites")

    print(counts)
    mask_chrom1 = mask[:counts[0]]

    mask_chrom2 = mask[counts[0]:counts[0]+counts[1]]
    
    mask_chrom3 = mask[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
    
    print(len(mask_chrom1))
    print(len(mask_chrom2))
    print(len(mask_chrom3))
    

    chrom1_mut_num = counts[0] - np.sum(mask_chrom1 != True)
    chrom2_mut_num = counts[1] - np.sum(mask_chrom2 != True)
    chrom3_mut_num = counts[2] - np.sum(mask_chrom3 != True)

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

    arr_chrom1 = pos[chrom == unique_chrom[0]][mask_chrom1]
    #arr_chrom1 = ts_chroms[0].sites_position[mask_chrom1] #array of site positions
    pairwise_distances_chrom1 = abs(arr_chrom1[:, None] - arr_chrom1) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom1 = pairwise_distances_chrom1[np.triu_indices_from(pairwise_distances_chrom1, k=1)]

    chrom1_scaled_ld = np.multiply(chrom1_ld, pairwise_distances_chrom1)

    ################Chrom2
    arr_chrom2 = pos[chrom == unique_chrom[1]][mask_chrom2]
    #arr_chrom2 = ts_chroms[1].sites_position[mask_chrom2] #array of site positions
    pairwise_distances_chrom2 = abs(arr_chrom2[:, None] - arr_chrom2) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom2 = pairwise_distances_chrom2[np.triu_indices_from(pairwise_distances_chrom2, k=1)]


    chrom2_scaled_ld = np.multiply(chrom2_ld, pairwise_distances_chrom2)

    #############Chrom3
    arr_chrom3 = pos[chrom == unique_chrom[2]][mask_chrom3]
    #arr_chrom3 = ts_chroms[2].sites_position[mask_chrom3] #array of site positions
    pairwise_distances_chrom3 = abs(arr_chrom3[:, None] - arr_chrom3) #broadcast subtraction to create matrix of pairwise distances between sites

    pairwise_distances_chrom3 = pairwise_distances_chrom3[np.triu_indices_from(pairwise_distances_chrom3, k=1)]


    chrom3_scaled_ld = np.multiply(chrom3_ld, pairwise_distances_chrom3)
    

    if len(mask_chrom1[mask_chrom1 != False]) != len(arr_chrom1) or len(mask_chrom2[mask_chrom2 != False]) != len(arr_chrom2) or len(mask_chrom3[mask_chrom3 != False]) != len(arr_chrom3):
        raise ValueError("Mask length does not equal pairwise distance array length")

    scaled_r2_norm = np.concatenate((chrom1_scaled_ld, chrom2_scaled_ld, chrom3_scaled_ld))

    return scaled_r2_norm


np.set_printoptions(legacy="1.21") #exclude dbtype from np arrays
summary_statistics = [] #Initialize list of summary statistics


vcf_path = "Bh_TRR356_srDNA_for_abc/ABC_regions_allFilters_3Mbp.vcf.gz"






callset = allel.read_vcf(
    vcf_path,
    fields=['calldata/GT', 'variants/REF', 'variants/ALT', 'variants/AA', 'variants/POS', 'variants/CHROM']
)


# Create GenotypeArray
gt = allel.GenotypeArray(callset['calldata/GT'])

print("Genotype shape:", gt.shape)
# Expected: (n_variants, 43, 1)
print(gt)

hap = gt[..., 0]

# Convert to HaplotypeArray
hap = allel.HaplotypeArray(hap)

ac = hap.count_alleles()
print(ac)


ref = callset['variants/REF']
alt = callset['variants/ALT'][:, 0]
aa = callset['variants/AA']

# True if REF is ancestral
ref_is_ancestral = (aa == ref)

# True if ALT is ancestral
alt_is_ancestral = (aa == alt)

# Start with ALT count
derived_counts = ac[:, 1].copy()

# If ALT is ancestral, derived allele is REF
derived_counts[alt_is_ancestral] = ac[alt_is_ancestral, 0]

valid = ref_is_ancestral | alt_is_ancestral
derived_counts = derived_counts[valid]

n = hap.n_haplotypes
sfs = np.bincount(derived_counts, minlength=n+1)


print(sfs)
print(len(sfs))

afs_entries = add_sfs_summary(afs = sfs, sample_size = 43, summary_statistics = summary_statistics) #12:26 SFS
print("Summary stat length = ",len(summary_statistics))
print(summary_statistics)

afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
summary_statistics.append(afs_quant[0]) #27 AFS quantile 0.1
summary_statistics.append(afs_quant[1]) #28 0.3
summary_statistics.append(afs_quant[2]) #29 0.5
summary_statistics.append(afs_quant[3]) #30 0.7
summary_statistics.append(afs_quant[4]) #31 0.9
afs_entries = []

pos = callset['variants/POS']
chrom = callset['variants/CHROM']
print("pos:", pos)
print("chrom:", chrom)

window_size = 100_000
results = []

for c in np.unique(chrom):
    
    mask = chrom == c
    
    pos_c = pos[mask]
    ac_c = ac[mask]
    
    # Compute filters on chromosome-specific array
    seg = ac_c.is_segregating()
    bi = ac_c.max_allele() == 1
    
    keep = seg & bi
    
    pos_c = pos_c[keep]
    ac_c = ac_c[keep]
    
    if len(pos_c) == 0:
        continue
    
    # Ensure sorted
    order = np.argsort(pos_c)
    pos_c = pos_c[order]
    ac_c = ac_c[order]
    
    tajd, windows, counts = allel.windowed_tajima_d(
        pos_c,
        ac_c,
        size=window_size,
        step=window_size
    )
    
    results.append(tajd)
print(results)
print(np.mean(results))

D_array = results
summary_statistics.append(np.nanmean(D_array)) #32 mean Tajima's D
summary_statistics.append(np.nanvar(D_array)) #33 variance of Tajima's D
summary_statistics.append(np.nanstd(D_array)) #34 std D
print(summary_statistics)

calculate_hamming(h = hap, ploidy = 1, summary_statistics = summary_statistics)
print(summary_statistics)

s, s_norm, mask = calculate_r2(g = hap, exclude_ac_below = exclude_ac_below)

print("s =", s)
print("s_norm:", s_norm)
print("mask:", mask)

unique_chrom, counts = np.unique(chrom, return_counts=True)

for c, n in zip(unique_chrom, counts):
    print(f"{c}: {n} sites")

print(counts)
print(counts[0])


r2 = get_rsq_per_chromosome(mask = mask, chrom = chrom, s = s)
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
summary_statistics.append(np.nanstd(r2_ge_1))
print(r2)
print(len(r2))
print(len(r2[r2 != 0]))
print(len(summary_statistics))


ild_all = get_ILD(mask = mask, chrom = chrom, s = s)
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
summary_statistics.append(np.nanstd(ild_ge_1))
print(len(summary_statistics))

print(ild_all)
print(len(ild_all))
print(len(ild_all[ild_all != 0]))

for c in np.unique(chrom):
    pos_c = pos[chrom == c]
    print(c, len(pos_c))
    print(pos_c)

arr_chrom1 = pos[chrom == unique_chrom[0]]
print("Arr_chrom:",arr_chrom1)
print(len(arr_chrom1))

scaled_r2 = calculate_Anderson_rsq(mask = mask, chrom = chrom, pos = pos, s = s)
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
print(summary_statistics)
print(len(summary_statistics))


r2_norm = get_rsq_norm_per_chromosome(mask = mask, chrom = chrom, r2_norm = s_norm)
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



ild_norm_all = get_ILD_norm(mask = mask, chrom = chrom, r2_norm = s_norm)  
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


scaled_r2_norm = calculate_Anderson_rsq_norm(mask, chrom = chrom, pos = pos, r2_norm = s_norm) 
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
print(summary_statistics)
print(len(summary_statistics))


n = 43
a1 = np.sum(1 / np.arange(1, n))
num_windows = 30 
windows = np.linspace(0, 3_000_000, num_windows + 1)
print(windows)
theta_pi = []
theta_w = []

results = []

n = 43  # number of haploid samples
print("n:", n)
a1 = np.sum(1 / np.arange(1, n))
print("a1:", a1)
print("chrom:", np.unique(chrom))
for c in np.unique(chrom):

    mask = chrom == c
    pos_c = pos[mask]
    ac_c = ac[mask]

    # Filter segregating + biallelic
    seg = ac_c.is_segregating()
    bi = ac_c.max_allele() == 1
    keep = seg & bi

    pos_c = pos_c[keep]
    ac_c = ac_c[keep]

    if len(pos_c) == 0:
        continue

    # Sort
    order = np.argsort(pos_c)
    pos_c = pos_c[order]
    ac_c = ac_c[order]

    # Haploid allele frequency
    p = ac_c[:, 1] / n
    pi_per_site = 2 * p * (1 - p)

    # ---- WINDOW DEFINITION FROM VCF POSITIONS ----
    start = pos_c.min()
    end = pos_c.max()

    windows = np.arange(start, end + window_size, window_size)

    stat_per_window = []

    for left, right in zip(windows[:-1], windows[1:]):

        loc = (pos_c >= left) & (pos_c < right)
        S = np.sum(loc)

        if S == 0:
            stat_per_window.append(np.nan)
            continue

        theta_pi = np.sum(pi_per_site[loc])
        theta_w = S / a1

        if theta_pi > 0:
            stat = (theta_pi - theta_w) / theta_pi
        else:
            stat = np.nan

        stat_per_window.append(stat)

    results.append(np.array(stat_per_window))
print("norm_Taj:", results)
print(np.mean(results))

summary_statistics.append(np.nanmean(results)) #102-103 normalized Tajima's D
summary_statistics.append(np.nanstd(results))

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

import pandas as pd
df = pd.DataFrame(summary_statistics).T
df.to_csv("observed_sum_stats.csv", index=False)
