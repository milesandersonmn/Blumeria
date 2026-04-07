# Load the eSMC2 library
library(eSMC2)

##############################
# Input & param
##############################
mu         = 4 * 10^-7       # mutation rate per bp per generation for Bgt: between 4 × 10−7 and 5 × 10−7
r          = 5e-8            # recombination rate per bp per generation (28 to 46 cM per 1MB, 2.8 * to 4.6 * 10^-7, 4 clonal phases = 5.6 to 9.2 * 10^-8)
nStates    = 30              # hidden states in HMM
##############################

# ----------- file locations (hidden from output) -----------------
data_dir   <- "~/PhD/Blumeria/SMBC/"
file_mhs   <- "Bh_TRR356_43_Bgt_TUM1_repmasked.chr_08_subs_1Mbto9Mb_3ind.mhs"

Os=Get_real_data(filename = file_mhs ,
                 path     = data_dir ,
                 M        = 6,
                 delim    = "\t")

### check Os
print(dim(Os)) ### [1]    8 1518


Os <- Os[c(1,3,5,7,8),] #Keep only haploids
Os[,1:10]
print(dim(Os))
#Os <- lapply(1:nrow(Os), function(i) as.list(Os[i, ]))

####
SMBC_results <- SMBC(
  O = Os,
  n      = nStates,
  rho    = r / mu,
  alpha  = 1.7,
  pop = F,
  B      = T, ### I want to estimate alpha
  ploidy = 1,
  NC     = 1,
  mu_real = mu,
  M_a = 3,
  #LH_opt = T
)


