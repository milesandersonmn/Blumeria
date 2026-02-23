#install.packages("abcrf")
library(abcrf)
#install.packages("pheatmap")
library(pheatmap)
#install.packages("reshape2")
library(reshape2)
library(ggplot2)

setwd("~/PhD/Blumeria/")

summary_table <- read.csv("summary_stats/growth/summary_statistics_contraction_mu5e-07_rho3.37e-07_2860_2900_unfoldedSFS.csv", header = TRUE)
#summary_table2 <- read.csv("summary_stats/growth/summary_statistics_constant_mu5e-07_rho3.37e-07_2860_2900_unfoldedSFS.csv", header = TRUE)



#names(summary_table2) <- names(summary_table)
summary_table <- rbind(summary_table, summary_table2)
#summary_table <- subset(summary_table, select = -V61)
#tail(summary_table)
txt <- summary_table[ ,-c(2:11)]
modindex <- as.factor(txt$X0)

sumsta <- txt[,-1]

header <- c( "modindex", 
            "fSFS1", "fSFS2", "fSFS3", 
            "fSFS4", "fSFS5", "fSFS6", "fSFS7", "fSFS8",
            "fSFS9", "fSFS10", "fSFS11", "fSFS12", "fSFS13",
            "fSFS14", "fSFS15", "AFS_q0.1", "AFS_q0.3", "AFS_q0.5",
            "AFS_q0.7", "AFS_q0.9", "mean_D", "var_D", "std_D",
            "ham_q0.1", "ham_q0.3",
            "ham_q0.5", "ham_q0.7",
            "ham_q0.9", "mean_ham", "std_ham",
            #"var_ham", "homo_q0.1", "homo_q0.3", "homo_q0.5", "homo_q0.7", "homo_q0.9", 
            #"mean_homo", "var_homo", "std_homo", 
            "r2_q0.1", "r2_q0.3", "r2_q0.5", "r2_q0.7", "r2_q0.9", "r2_q0.95", "r2_q0.99",
            "mean_r2", "var_r2", "std_r2", "r2_ge_1",
            "ILD_q0.1", "ILD_q0.3", "ILD_q0.5", "ILD_q0.7", "ILD_q0.9", "ILD_q0.95", "ILD_q0.99",
            "mean_ILD", "var_ILD", "std_ILD", "ILD_ge_1",
            "AndR2_q0.1", "AndR2_q0.3", "AndR2_q0.5", "AndR2_q0.7", "AndR2_q0.9", "AndR2_q0.95", "AndR2_q0.99",
            "mean_AndR2", "var_AndR2", "std_AndR2",
            "norm_r2_q0.1", "norm_r2_q0.3", "norm_r2_q0.5", "norm_r2_q0.7", "norm_r2_q0.9", "norm_r2_q0.95", "norm_r2_q0.99",
            "mean_norm_r2", "var_norm_r2", "std_norm_r2", "norm_r2_ge_1",
            "norm_ILD_q0.1", "norm_ILD_q0.3", "norm_ILD_q0.5", "norm_ILD_q0.7", "norm_ILD_q0.9", "norm_ILD_q0.95", "norm_ILD_q0.99",
            "mean_norm_ILD", "var_norm_ILD", "std_norm_ILD", "norm_ILD_ge_1",
            "norm_AndR2_q0.1", "norm_AndR2_q0.3", "norm_AndR2_q0.5", "norm_AndR2_q0.7", "norm_AndR2_q0.9", "norm_AndR2_q0.95", "norm_AndR2_q0.99",
            "mean_norm_AndR2", "var_norm_AndR2", "std_norm_AndR2",
            "mean_norm_Taj_D", "std_norm_Taj_D",
            "LDFS_0.1", "LDFS_0.2", "LDFS_0.3", "LDFS_0.4", "LDFS_0.5", "LDFS_0.6", "LDFS_0.7", "LDFS_0.8", "LDFS_0.9", "LDFS_1",
            "LDFS_diff",
            "ILDFS_0.1", "ILDFS_0.2", "ILDFS_0.3", "ILDFS_0.4", "ILDFS_0.5", "ILDFS_0.6", "ILDFS_0.7", "ILDFS_0.8", "ILDFS_0.9", "ILDFS_1",
            "ILDFS_diff",
            "norm_LDFS_0.1", "norm_LDFS_0.2", "norm_LDFS_0.3", "norm_LDFS_0.4", "norm_LDFS_0.5", "norm_LDFS_0.6", "norm_LDFS_0.7", "norm_LDFS_0.8", "norm_LDFS_0.9", "norm_LDFS_1",
            "norm_LDFS_diff",
            "norm_ILDFS_0.1", "norm_ILDFS_0.2", "norm_ILDFS_0.3", "norm_ILDFS_0.4", "norm_ILDFS_0.5", "norm_ILDFS_0.6", "norm_ILDFS_0.7", "norm_ILDFS_0.8", "norm_ILDFS_0.9", "norm_ILDFS_1",
            "norm_ILDFS_diff")
length(header)


data1 <- data.frame(modindex, sumsta)
colnames(data1) <- header

cv_Andersons_rsq <- data1$std_AndR2 / data1$mean_AndR2
cv_ILD <- data1$std_ILD / data1$mean_ILD
cv_rsq <- data1$std_r2 / data1$mean_r2
cv_norm_Andersons_rsq <- data1$std_norm_AndR2 / data1$mean_norm_AndR2
cv_norm_ILD <- data1$std_norm_ILD / data1$mean_norm_ILD
cv_norm_rsq <- data1$std_norm_r2 / data1$mean_norm_r2


data1 <- cbind(data1, cv_Andersons_rsq, cv_ILD, cv_rsq, cv_norm_Andersons_rsq, cv_norm_ILD, cv_norm_rsq)
#data1 <- data1[, -c(23,32,39,47,55,63)]
#data1 <- data1[ ,-c(2:6,14:20)]
###Remove Singletons
#data1 <- data1[ ,-c(2)]
data1 <- data1[, !grepl("homo", colnames(data1))]
model.rf1 <- abcrf(modindex~., data = data1, lda = FALSE)
model.rf1
conf_mat <- model.rf1$model.rf$confusion.matrix
err.abcrf(model.rf1, data1)

pheatmap(conf_mat[,1:5], 
         display_numbers = TRUE, 
         #color = scales::div_gradient_pal(low = "blue",
          #                                mid = "yellow",
           #                               high="red")(seq(0,1,
            #                                                 length.out = max(conf_mat))),
         cluster_rows = FALSE, 
         cluster_cols = FALSE,
         border_color = FALSE,
         number_color = "black",
        
         main = "Confusion Matrix Heatmap",
         labels_col = c("1.9","1.7","1.5","1.3","1.1"),
         labels_row = c("1.9","1.7","1.5","1.3","1.1"))


quartz(height = 5)
plot(1:20,1:20)
plot(model.rf1, data1, n.var = 35)
model.rf1$model.rf$variable.importance
sort(model.rf1$model.rf$variable.importance)

#head(summary_table)
#obs <- txt[c(20000,43000,65000,83000,110000, 16000, 45000, 67000, 90000, 120000), -1]

obs <- read.csv("observed_sum_stats.csv", header = TRUE)
#obs <- obs[, -c(1:11)]
#obs <- obs[, -c(23,27,35,43,51,59,67)]
colnames(obs) <- header

cv_Andersons_rsq <- obs$std_AndR2 / obs$mean_AndR2
cv_ILD <- obs$std_ILD / obs$mean_ILD
cv_rsq <- obs$std_r2 / obs$mean_r2
cv_norm_Andersons_rsq <- obs$std_norm_AndR2 / obs$mean_norm_AndR2
cv_norm_ILD <- obs$std_norm_ILD / obs$mean_norm_ILD
cv_norm_rsq <- obs$std_norm_r2 / obs$mean_norm_r2


obs <- cbind(obs, cv_Andersons_rsq, cv_ILD, cv_rsq, cv_norm_Andersons_rsq, cv_norm_ILD, cv_norm_rsq)
names(obs) <- names(subset(data1, select = -modindex))
#obs <- obs[, -c(1:5, 13:19)]

prediction <- predict(model.rf1, obs, data1)
prediction
ggplot(data1, aes(x = modindex, y = std_AndR2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_AndR2", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = mean_AndR2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of mean_AndR2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = std_r2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_r2", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = std_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_ILD", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = mean_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of mean_ILD", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_ILD", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_norm_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_ILD", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_norm_rsq)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_rsq)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_norm_Andersons_rsq)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_Andersons_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = cv_Andersons_rsq)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_Andersons_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = norm_AndR2_q0.9)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_Andersons_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = mean_D)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_Andersons_r2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = fSFS1)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of CV_norm_Andersons_r2", x = "model_index", y = "mean")




curve(25000*exp(0.000001*x),
      from = 0, to = 100000,
      xlab = "Generations")
      