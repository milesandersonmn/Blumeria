import msprime
import numpy as np
import scipy
import allel 

#--------------------
#Helper Functions
#--------------------

#######Instantiate a beta coalescent demography with msprime
def instantiate_ts(alpha, initial_size, growth_rate, sample_size, rate_map, ploidy = 1):
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


######Calculate pairwise r-squared for entire genome with allele count pruning

def calculate_r2(mts, exclude_ac_below, sample_size):
    
    g = mts.genotype_matrix()
    g = allel.HaplotypeArray(g)

    # Count *all* alleles
    ac_all = g.count_alleles(max_allele=None)

    # Keep only sites where max allele index == 1 (i.e. only 0/1 present)
    biallelic_mask = ac_all.max_allele() == 1

    # Apply both filters
    mask = biallelic_mask & (ac_all[:, 1] > exclude_ac_below) & (ac_all[:, 1] <= sample_size - exclude_ac_below)

    gn_filt = g[mask].astype(np.float64)


    

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
    

    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

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
    

    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

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

    return r2_norm

######Get ILD

def get_ILD(mask, ts_chroms, s):
    
    mask_chrom1 = mask[:ts_chroms[0].num_sites]

    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]

    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    # Chromosome index boundaries
    c1 = chrom1_mut_num
    c2 = chrom1and2_mut_num
    c3 = total_mut_num   # total mutation count



    # ---- Only take each inter-chromosome comparison once ---- #

    # Chrom1 vs Chrom2  (upper-right block)
    chrom1_2 = s[0:c1, c1:c2].ravel()

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
 
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)

    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    # Chromosome index boundaries
    c1 = chrom1_mut_num
    c2 = chrom1and2_mut_num
    c3 = total_mut_num   # total mutation count


    # ---- Only take each inter-chromosome comparison once ---- #
 
    # Chrom1 vs Chrom2  (upper-right block)
    chrom1_2 = r2_norm[0:c1, c1:c2].ravel()

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
    

    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)


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
    
 
    

    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)


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

######Fold 2SFS

def fold_2sfs(sfs_2d, n):
    size = n // 2 + 1
    sfs_2d_folded = np.zeros((size, size), dtype=np.int64)
    for i in range(n + 1):
        for j in range(n + 1):
            fi = min(i, n - i)
            fj = min(j, n - j)
            sfs_2d_folded[fi, fj] += sfs_2d[i, j]
    return sfs_2d_folded


######Compute hi_lo PMC

def compute_hiloPMI(sfs_2d_folded, mac, ic):
    eta_lo = np.sum((mac >= 1) & (mac < ic)) #/ len(mac)
    eta_hi = np.sum(mac >= ic) #/ len(mac)

    total_pairs = sfs_2d_folded[1:, :].sum()
    eta_hilo = sfs_2d_folded[1:ic, ic:].sum() #/ total_pairs

    print("double sum:", sfs_2d_folded[1:ic, ic:])
    print(f"eta_lo: {eta_lo:.4f}, eta_hi: {eta_hi:.4f}, eta_hilo: {eta_hilo:.4f}")
    print(f"lo*hi: {eta_lo * eta_hi:.4f}")

    return np.log(eta_hilo / (eta_lo * eta_hi))


######Windowed test hiloPMC
def compute_window_hiloPMI(mts, n, window_start, window_end, ic=2):
    # --- 1. Get biallelic genotypes in window ---
    import itertools
    biallelic_genotype_matrix = []

    for var in mts.variants():
        if not (window_start <= var.site.position < window_end):
            continue
        alleles = [a for a in var.alleles if a is not None]
        if len(alleles) == 2:
            biallelic_genotype_matrix.append(var.genotypes.copy())

    if len(biallelic_genotype_matrix) < 2:
        return None  # need at least 2 sites for a pair
    # Also count monomorphic sites
    # Approximate: total sites in window = window_end - window_start
    # Or use the actual site count if you have it
    total_sites_in_window = window_end - window_start  # every base is a site


    G = np.array(biallelic_genotype_matrix, dtype=np.int8)

    # --- 2. Compute MAC directly ---
    dac = G.sum(axis=1)
    mac = np.minimum(dac, n - dac)
    mac = mac[mac > 0]  # exclude monomorphic

    if len(mac) < 2:
        return None

    # --- 3. Build folded 2SFS directly from MAC, upper triangle only ---
    size = n // 2 + 1
    sfs_2d_folded = np.zeros((size, size), dtype=np.int64)

    num_sites = len(mac)
    for i, j in itertools.combinations(range(num_sites), 2):
        sfs_2d_folded[mac[i], mac[j]] += 1

    # --- 4. Compute hiloPMI ---
    total_pairs = sfs_2d_folded[1:, 1:].sum()
    if total_pairs == 0:
        return None

    #eta_lo   = np.sum((mac >= 1) & (mac < ic)) 
    #eta_hi   = np.sum(mac >= ic)
    #eta_hilo = sfs_2d_folded[1:ic, ic:].sum() 

    #eta_lo   = np.sum((mac >= 1) & (mac < ic)) / len(mac)
    #eta_hi   = np.sum(mac >= ic) / len(mac)
    #eta_hilo = sfs_2d_folded[1:ic, ic:].sum() / total_pairs
    total_pairs = total_sites_in_window * (total_sites_in_window - 1) // 2

    # eta_lo, eta_hi: fraction of ALL sites (including monomorphic)
    eta_lo   = np.sum((mac >= 1) & (mac < ic)) / total_sites_in_window
    eta_hi   = np.sum(mac >= ic) / total_sites_in_window

    # eta_hilo: fraction of ALL pairs
    eta_hilo = sfs_2d_folded[1:ic, ic:].sum() / total_pairs
    if eta_lo == 0 or eta_hi == 0 or eta_hilo == 0:
        return None
    hilo_PMI = np.log(eta_hilo/(eta_lo * eta_hi))
    return hilo_PMI, eta_hilo, eta_lo, eta_hi



def get_weighted_rsq_stats_per_chromosome(mask, ts_chroms, s, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):

    def weighted_quantiles(values, weights, quantiles):
        sorter = np.argsort(values)
        sorted_vals = values[sorter]
        sorted_weights = weights[sorter]
        cumulative_weights = np.cumsum(sorted_weights)
        cumulative_weights /= cumulative_weights[-1]
        return np.interp(quantiles, cumulative_weights, sorted_vals)

    def weighted_stats(r2_adj, dist_adj, quantiles):
        weights = 1 / dist_adj
        weighted_mean = np.sum(weights * r2_adj) / np.sum(weights)
        V1 = np.sum(weights)
        V2 = np.sum(weights**2)
        weighted_var = (V1 / (V1**2 - V2)) * np.sum(weights * (r2_adj - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_var)
        wq = weighted_quantiles(r2_adj, weights, quantiles)
        return weighted_mean, weighted_std, wq

    # Split mask per chromosome
    mask_chrom1 = mask[:ts_chroms[0].num_sites]
    mask_chrom2 = mask[ts_chroms[0].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites]
    mask_chrom3 = mask[ts_chroms[0].num_sites+ts_chroms[1].num_sites:ts_chroms[0].num_sites+ts_chroms[1].num_sites+ts_chroms[2].num_sites]

    # Mutation counts per chromosome
    chrom1_mut_num = ts_chroms[0].num_sites - np.sum(mask_chrom1 != True)
    chrom2_mut_num = ts_chroms[1].num_sites - np.sum(mask_chrom2 != True)
    chrom3_mut_num = ts_chroms[2].num_sites - np.sum(mask_chrom3 != True)
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    total_mut_num = chrom1_mut_num + chrom2_mut_num + chrom3_mut_num

    # Slice r2 matrix and compute adjacent distances per chromosome
    all_r2 = []
    all_dist = []
    for ts_chrom, mask_chrom, start, end in zip(
        ts_chroms,
        [mask_chrom1, mask_chrom2, mask_chrom3],
        [0, chrom1_mut_num, chrom1and2_mut_num],
        [chrom1_mut_num, chrom1and2_mut_num, total_mut_num]
    ):
        ld_mat = s[start:end, start:end]
        n = ld_mat.shape[0]
        adj_idx = (np.arange(n-1), np.arange(1, n))
        all_r2.append(ld_mat[adj_idx])

        # Compute adjacent distances directly from site positions
        positions = ts_chrom.sites_position[mask_chrom]
        all_dist.append(np.diff(positions))

    # Combine across chromosomes and filter zero distances
    all_r2 = np.concatenate(all_r2)
    all_dist = np.concatenate(all_dist)
    valid = all_dist > 0
    all_r2, all_dist = all_r2[valid], all_dist[valid]

    mean, std, wq = weighted_stats(all_r2, all_dist, quantiles)
    print("weighted average r2:", mean)
    print("weighted std r2:", std)
    print("weighted quantiles r2:", wq)

    return {
        'weighted_mean': mean,
        'weighted_std': std,
        'weighted_quantiles': dict(zip(quantiles, wq))
    }
