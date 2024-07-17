load("CFA_Pt.RData")
library(bruceR)
library(lavaan)

all_MLLMs_data <- import("MLLMs_score/MLLMs_acc_pt_score.xlsx")
names(all_MLLMs_data)
MLLMs_data <- all_MLLMs_data[,2:19]
MLLMs_data %>% na.omit() %>% apply(2,scale) %>% as.data.frame() -> z_score_data  # normalize by column, zscore

GIA_scores <- lavPredict(fit_behav2, newdata = MLLMs_data)
GIA_scores[,6]

cor.test(all_MLLMs_data$total_acc, GIA_scores[,6])
GLM_summary(lm(all_MLLMs_data$total_acc~GIA_scores[,6]))
