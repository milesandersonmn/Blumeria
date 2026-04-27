#install.packages("abcrf")
library(abcrf)
#install.packages("pheatmap")
library(pheatmap)
#install.packages("reshape2")
library(reshape2)
library(ggplot2)

setwd("~/PhD/Blumeria/")

summary_table <- read.csv("summary_stats/summary_statistics_mu5e-07_rho3.37e-07_2860_2900_unfoldedSFS.csv", header = TRUE)
#summary_table2 <- read.csv("summary_stats/growth/summary_statistics_constant_mu5e-07_rho3.37e-07_2860_2900_unfoldedSFS.csv", header = TRUE)



#names(summary_table2) <- names(summary_table)
#summary_table <- rbind(summary_table, summary_table2)
#summary_table <- subset(summary_table, select = -V61)
#tail(summary_table)
txt <- summary_table[ ,-c(2:12)]
modindex <- as.factor(txt$X0)

sumsta <- txt[,-1]

header <- c( "modindex", 
            "fSFS1", "fSFS2", "fSFS3", 
            "fSFS4", "fSFS5", "fSFS6", "fSFS7", "fSFS8",
            "fSFS9", "fSFS10", "fSFS11", "fSFS12", "fSFS13",
            "fSFS14", "fSFS15", "AFS_q0.1", "AFS_q0.3", "AFS_q0.5", "AFS_q0.7", "AFS_q0.9", 
            "mean_D", "var_D", "std_D",
            "ham_q0.1", "ham_q0.3",
            "ham_q0.5", "ham_q0.7",
            "ham_q0.9", "mean_ham", "std_ham", "var_ham",
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
            "norm_ILDFS_diff"
            ,
            "hiloPMI", "D_norm_hiloPMI", "window_D_norm_hiloPMI"
            )
length(header)


data1 <- data.frame(modindex, sumsta)
colnames(data1) <- header

cv_Andersons_rsq <- data1$std_AndR2 / data1$mean_AndR2
cv_ILD <- data1$std_ILD / data1$mean_ILD
cv_rsq <- data1$std_r2 / data1$mean_r2
cv_norm_Andersons_rsq <- data1$std_norm_AndR2 / data1$mean_norm_AndR2
cv_norm_ILD <- data1$std_norm_ILD / data1$mean_norm_ILD
cv_norm_rsq <- data1$std_norm_r2 / data1$mean_norm_r2
cv_norm_D <- data1$std_norm_Taj_D / data1$mean_norm_Taj_D
#single_double_ratio <- data1$fSFS1 / data1$fSFS2
#ZnS <- data1$mean_ILD - data1$mean_r2
#psi <- data1$mean_r2 / data1$mean_ILD
#psi_ratio <- data1$psi / data1$single_double_ratio

#data1 <- cbind(data1, single_double_ratio)
#data1 <- cbind(data1, psi_ratio)
#data1 <- cbind(data1, psi)
#data1 <- cbind(data1, ZnS)
data1 <- cbind(data1, cv_Andersons_rsq, cv_ILD, cv_rsq, cv_norm_Andersons_rsq, cv_norm_ILD, cv_norm_rsq, cv_norm_D)

ggplot(data1, aes(x = modindex, y = AFS_q0.5)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_AndR2", x = "model_index", y = "std")


data1 <- data1[, !grepl("hiloPMI", colnames(data1))]

data1 <- data1[, !grepl("And", colnames(data1))]

library(dplyr)

data_balanced <- data1 %>%
  group_by(modindex) %>%
  slice_sample(n = 20000) %>%
  ungroup()

data1 <- data_balanced
model.rf1 <- abcrf(modindex~., data = data1, lda = FALSE)
model.rf1
conf_mat <- model.rf1$model.rf$confusion.matrix
#err.abcrf(model.rf1, data1)

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
        
         main = "Beta-Coalescent Confusion Matrix",
         labels_col = c("1.9","1.7","1.5","1.3","1.1"),
         labels_row = c("1.9","1.7","1.5","1.3","1.1"))


quartz(height = 5)
plot(1:20,1:20)
plot(model.rf1, data1, n.var = 10) 
model.rf1$model.rf$variable.importance
sort(model.rf1$model.rf$variable.importance)


obs <- read.csv("observed_sum_stats.csv", header = TRUE)
colnames(obs) <- header[-1]
obs <- obs[,1:141]

cv_Andersons_rsq <- obs$std_AndR2 / obs$mean_AndR2
cv_ILD <- obs$std_ILD / obs$mean_ILD
cv_rsq <- obs$std_r2 / obs$mean_r2
cv_norm_Andersons_rsq <- obs$std_norm_AndR2 / obs$mean_norm_AndR2
cv_norm_ILD <- obs$std_norm_ILD / obs$mean_norm_ILD
cv_norm_rsq <- obs$std_norm_r2 / obs$mean_norm_r2
cv_norm_D <- obs$std_norm_Taj_D / obs$mean_norm_Taj_D
#single_double_ratio_obs <- obs$fSFS1 / obs$fSFS2
#ZnS_obs <- obs$mean_ILD - obs$mean_r2
#psi_obs <- obs$mean_r2 / obs$mean_ILD
#psi_ratio_obs <- psi_obs / single_double_ratio




obs <- cbind(obs, cv_Andersons_rsq, cv_ILD, cv_rsq, cv_norm_Andersons_rsq, cv_norm_ILD, cv_norm_rsq, cv_norm_D)
obs <- obs[, !grepl("fSFS", colnames(obs))]
obs <- obs[, !grepl("AFS", colnames(obs))]
obs <- obs[, !grepl("_D", colnames(obs))]
obs <- obs[, !grepl("And", colnames(obs))]


#names(obs) <- names(subset(data1, select = -modindex))
#obs <- obs[, -c(1:5, 13:19)]

prediction <- predict(model.rf1, obs, data1)
prediction

library(ggplot2)

library(ggplot2)

# Inspect the structure once if you're unsure of the slot names:
# str(prediction)

# Pull values directly from the prediction object
votes_vec    <- as.numeric(prediction$vote[1, ])   # row 1 = first (and only) observation
post_proba   <- prediction$post.prob[1]
selected_idx <- which.max(votes_vec)               # winning model = most votes

# Map model indices -> alpha values (must be in the same order as prediction$vote columns)
alpha_values <- c("1.9", "1.7", "1.5", "1.3", "1.1")

votes <- data.frame(
  model    = factor(alpha_values, levels = alpha_values),
  votes    = votes_vec,
  selected = seq_along(alpha_values) == selected_idx
)

p <- ggplot(votes, aes(x = model, y = votes, fill = selected)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = votes), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("TRUE" = "#534AB7", "FALSE" = "grey70"),
                    guide = "none") +
  labs(
    x = expression(paste("Model ", alpha, " value")),
    y = "Number of model votes",
    title = "ABC Model Choice (Full Summary Statistics)",
    subtitle = bquote("Selected: "*alpha*" = "*.(alpha_values[selected_idx])*
                        "   |   Posterior probability = "*.(sprintf("%.3f", post_proba)))
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "grey30", size = 10, margin = margin(b = 10)),
    axis.text     = element_text(color = "black"),
    axis.line     = element_line(linewidth = 0.4)
  )
p
ggsave("abc_rf_selection_fullSFS.png", plot = p,
       width = 5.5, height = 4, dpi = 600, bg = "white")
####################################
votes <- data.frame(
  model = factor(c("1.9", "1.7", "1.5", "1.3", "1.1"),
                 levels = c("1.9", "1.7", "1.5", "1.3", "1.1")),  # preserve order
  votes = c(502, 452, 46, 0, 0)
)
votes$selected <- votes$model == "1.9"
post_proba <- 0.6257167

p <- ggplot(votes, aes(x = model, y = votes, fill = selected)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = votes), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("TRUE" = "#534AB7", "FALSE" = "grey70"),
                    guide = "none") +
  labs(
    x = expression(paste("Model ", alpha, " value")),
    #x = "Model",
    y = "Number of model votes",
    title = "ABC Model Choice (Full Summary Statistics)",
    subtitle = sprintf("Selected: 1.9   |   Posterior probability = %.3f", post_proba)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "grey30", size = 10, margin = margin(b = 10)),
    axis.text = element_text(color = "black"),
    axis.line = element_line(linewidth = 0.4)
  )
ggsave("abc_rf_selection_fullSFS.pdf", width = 5.5, height = 4, device = cairo_pdf)
# Also save a high-res PNG/TIFF if the journal needs raster:
ggsave("abc_rf_selection_fullSFS.tiff", width = 5.5, height = 4, dpi = 600, compression = "lzw")

ggsave(
  filename = "abc_rf_selection_fullSFS.png",
  plot     = p,
  width    = 5.5,        # inches — fits a single column nicely
  height   = 4,
  dpi      = 600,        # 300 is acceptable, 600 is safer for print
  bg       = "white"     # avoids a transparent background in the PDF
)
#######################
data1_no_singletons <- data1[, !grepl("fSFS1", colnames(data1))]
data1 <- data1[, !grepl("fSFS", colnames(data1))]
data1 <- data1[, !grepl("AFS", colnames(data1))]
data1 <- data1[, !grepl("_D", colnames(data1))]

model.rf1_no_singletons <- abcrf(modindex~., data = data1_no_singletons, lda = FALSE)
model.rf1_no_singletons
conf_mat_no_singletons <- model.rf1_no_singletons$model.rf$confusion.matrix


pheatmap(conf_mat_no_singletons[,1:5], 
         display_numbers = TRUE, 
         #color = scales::div_gradient_pal(low = "blue",
         #                                mid = "yellow",
         #                               high="red")(seq(0,1,
         #                                                 length.out = max(conf_mat))),
         cluster_rows = FALSE, 
         cluster_cols = FALSE,
         border_color = FALSE,
         number_color = "black",
         
         main = "Beta-Coalescent Confusion Matrix (No Singletons)",
         labels_col = c("1.9","1.7","1.5","1.3","1.1"),
         labels_row = c("1.9","1.7","1.5","1.3","1.1"))


quartz(height = 5)
plot(1:20,1:20)
plot(model.rf1, data1, n.var = 10) 
model.rf1$model.rf$variable.importance
sort(model.rf1$model.rf$variable.importance)

ggplot(data1, aes(x = modindex, y = fSFS1)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of mean_AndR2", x = "model_index", y = "mean")






#curve(25000*exp(0.000001*x),
#      from = 0, to = 10000,
#      xlab = "Generations")
      