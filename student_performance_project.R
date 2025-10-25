# student_performance_project.R
# Author: Mayesha Maliha Proma
#
# What this script does
# - Loads the UCI student performance (math) data
# - Explores class balance, grade distribution, correlations
# - Trains and tunes two models: Random Forest and XGBoost
# - Compares metrics and saves figures + a CSV with results
#
# How to run
# - Open R (or RStudio) in this folder
# - Run: source("student_performance_project.R")
# - See outputs/ for saved images and metrics
#
# Reproducibility
# - Fixed random seed
# - Warnings show immediately

set.seed(123)
options(warn = 1)

# -------------------------
# Packages
# -------------------------
# Install missing packages quietly, then load them.
install_if_missing <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      install.packages(p, repos = "https://cloud.r-project.org")
    }
  }
}

install_if_missing(c(
  "tidyverse","tidymodels","janitor","lubridate","stringr",
  "ranger","xgboost","vip","knitr","rmarkdown"
))

suppressPackageStartupMessages({
  library(tidyverse)   # data wrangling + ggplot2
  library(tidymodels)  # recipes, models, resampling, tuning
  library(janitor)     # clean_names()
  library(lubridate)   # dates (handy, optional)
  library(stringr)     # string tools
  library(ranger)      # fast Random Forest
  library(xgboost)     # boosted trees
  library(vip)         # variable importance (optional)
})

# -------------------------
# Folders
# -------------------------
# Keep output files organized.
dir.create("data",    showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Data helper
# -------------------------
# Use a local CSV if present.
# Otherwise look for the official ZIPs, or download from UCI.
get_student_csv <- function() {
  # 1) Local CSV next to this script or under data/
  candidates <- c("student-mat.csv", "data/student-mat.csv")
  for (p in candidates) if (file.exists(p)) return(p)

  # Helper to extract the specific CSV from a ZIP
  extract_csv <- function(zpath) {
    files <- unzip(zpath, list = TRUE)$Name
    csv_name <- files[grepl("student-mat\\.csv$", files)]
    if (length(csv_name) == 0) stop("student-mat.csv not found in: ", zpath)
    unzip(zpath, files = csv_name[1], exdir = "data")
    file.path("data", basename(csv_name[1]))
  }

  # 2) Local ZIPs (outer student.zip or nested within student+performance.zip)
  if (file.exists("student.zip")) return(extract_csv("student.zip"))
  if (file.exists("data/student.zip")) return(extract_csv("data/student.zip"))

  if (file.exists("student+performance.zip")) {
    dir.create("data_raw", showWarnings = FALSE)
    unzip("student+performance.zip", exdir = "data_raw")
    inner <- list.files("data_raw", pattern = "student\\.zip$", recursive = TRUE, full.names = TRUE)
    if (length(inner) == 0) stop("Inner student.zip not found in outer zip")
    return(extract_csv(inner[1]))
  }
  if (file.exists("data/student+performance.zip")) {
    dir.create("data_raw", showWarnings = FALSE)
    unzip("data/student+performance.zip", exdir = "data_raw")
    inner <- list.files("data_raw", pattern = "student\\.zip$", recursive = TRUE, full.names = TRUE)
    if (length(inner) == 0) stop("Inner student.zip not found in outer zip")
    return(extract_csv(inner[1]))
  }

  # 3) Download the official UCI archive as a last resort
  zip_url  <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
  zip_file <- file.path("data", "student.zip")
  download.file(zip_url, destfile = zip_file, mode = "wb", quiet = TRUE)
  extract_csv(zip_file)
}

csv_path <- get_student_csv()
message("Using CSV: ", csv_path)

# -------------------------
# Load and label
# -------------------------
# - UCI file uses ';' as delimiter
# - Rename columns to snake_case
# - Create pass_flag target (g3 >= 10 is pass)
raw <- readr::read_delim(csv_path, delim = ";", show_col_types = FALSE) %>%
  clean_names()

dat <- raw %>%
  mutate(pass_flag = factor(if_else(g3 >= 10, "pass", "not_pass"),
                            levels = c("pass","not_pass")))

# -------------------------
# EDA (saved to outputs/)
# -------------------------

# Class balance
p_bal <- dat %>%
  count(pass_flag) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(pass_flag, pct, label = scales::percent(pct, accuracy = 0.1))) +
  geom_col() + geom_text(vjust = -0.5) +
  labs(title = "Class Balance (pass vs not_pass)", x = NULL, y = "Proportion")
ggsave("outputs/01_class_balance.png", p_bal, width = 6, height = 4, dpi = 150)

# Final grade histogram
p_hist <- ggplot(dat, aes(g3, fill = pass_flag)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.6) +
  labs(title = "Distribution of Final Grade (g3)", x = "g3", y = "Count")
ggsave("outputs/02_g3_hist.png", p_hist, width = 6, height = 4, dpi = 150)

# Numeric correlation heatmap
num_vars  <- dat %>% select(where(is.numeric))
cormat    <- suppressWarnings(cor(num_vars, use = "pairwise.complete.obs"))
cormat_df <- as_tibble(as.table(round(cormat, 2)), .name_repair = "minimal")
names(cormat_df) <- c("Var1","Var2","Corr")
p_cor <- cormat_df %>%
  ggplot(aes(Var1, Var2, fill = Corr)) +
  geom_tile() + scale_fill_gradient2(limits = c(-1,1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap", x = NULL, y = NULL)
ggsave("outputs/03_correlation_heatmap.png", p_cor, width = 7, height = 6, dpi = 150)

# -------------------------
# Modeling setup
# -------------------------
# - 80/20 split stratified by pass_flag
# - 5-fold CV for tuning
# - Recipe drops g3 to avoid leakage; imputes, encodes, normalizes
set.seed(123)
split <- initial_split(dat, prop = 0.8, strata = pass_flag)
train <- training(split)
test  <- testing(split)
cv    <- vfold_cv(train, v = 5, strata = pass_flag)

rec <- recipe(pass_flag ~ ., data = train) %>%
  step_rm(g3) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01, other = "__rare__") %>%
  step_novel(all_nominal_predictors(), new_level = "__new__") %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Common metrics
metrics <- metric_set(roc_auc, pr_auc, accuracy, sens, spec)

# -------------------------
# Model 1: Random Forest
# -------------------------
# Why RF
# - Strong baseline for mixed tabular data
# - Captures non-linearities and interactions
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>% add_model(rf_spec) %>% add_recipe(rec)

# Space-filling grid (mtry, min_n)
rf_params <- parameters(
  finalize(mtry(), train),
  min_n()
)
rf_grid <- grid_max_entropy(rf_params, size = 15)

rf_res <- tune_grid(rf_wf, resamples = cv, grid = rf_grid, metrics = metrics)
rf_best <- select_best(rf_res, metric = "roc_auc")
rf_final_wf <- finalize_workflow(rf_wf, rf_best) %>% fit(train)

# Test metrics
rf_pred <- predict(rf_final_wf, test, type = "prob") %>%
  bind_cols(predict(rf_final_wf, test),
            test %>% select(pass_flag))

rf_eval <- tibble(
  model   = "RandomForest",
  roc_auc = roc_auc(rf_pred, pass_flag, .pred_not_pass, event_level = "second")$.estimate,
  pr_auc  = pr_auc (rf_pred, pass_flag, .pred_not_pass, event_level = "second")$.estimate,
  acc     = accuracy(rf_pred, pass_flag, .pred_class)$.estimate,
  sensi   = sens(rf_pred, pass_flag, .pred_class, event_level = "second")$.estimate,
  speci   = spec(rf_pred, pass_flag, .pred_class, event_level = "second")$.estimate
)

# -------------------------
# Model 2: XGBoost
# -------------------------
# Why XGB
# - Often top performer on tabular problems
# - More knobs to tune; can squeeze extra accuracy
xgb_spec <- boost_tree(
  trees = 1000,
  learn_rate     = tune(),
  tree_depth     = tune(),
  mtry           = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(rec)

# Use sample_prop in the grid (maps to sample_size internally)
xgb_params <- parameters(
  learn_rate(),
  tree_depth(),
  finalize(mtry(), train),
  min_n(),
  loss_reduction(),
  sample_prop(range = c(0.5, 1.0))
)
xgb_grid <- grid_max_entropy(xgb_params, size = 20)

xgb_res <- tune_grid(xgb_wf, resamples = cv, grid = xgb_grid, metrics = metrics)
xgb_best <- select_best(xgb_res, metric = "roc_auc")
xgb_final_wf <- finalize_workflow(xgb_wf, xgb_best) %>% fit(train)

# Test metrics
xgb_pred <- predict(xgb_final_wf, test, type = "prob") %>%
  bind_cols(predict(xgb_final_wf, test),
            test %>% select(pass_flag))

xgb_eval <- tibble(
  model   = "XGBoost",
  roc_auc = roc_auc(xgb_pred, pass_flag, .pred_not_pass, event_level = "second")$.estimate,
  pr_auc  = pr_auc (xgb_pred, pass_flag, .pred_not_pass, event_level = "second")$.estimate,
  acc     = accuracy(xgb_pred, pass_flag, .pred_class)$.estimate,
  sensi   = sens(xgb_pred, pass_flag, .pred_class, event_level = "second")$.estimate,
  speci   = spec(xgb_pred, pass_flag, .pred_class, event_level = "second")$.estimate
)

# -------------------------
# Compare and save
# -------------------------
comparison <- bind_rows(rf_eval, xgb_eval)
readr::write_csv(comparison, "outputs/metrics_comparison.csv")
print(comparison)

# -------------------------
# Confusion matrix for the winner
# -------------------------
best_is_rf <- comparison$roc_auc[comparison$model == "RandomForest"] >=
              comparison$roc_auc[comparison$model == "XGBoost"]
best_pred  <- if (best_is_rf) rf_pred else xgb_pred

cm <- yardstick::conf_mat(best_pred, truth = pass_flag, estimate = .pred_class)
print(cm)

p_cm <- as.data.frame(cm$table) %>%
  ggplot(aes(Prediction, Truth, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix (Best Model)", x = "Prediction", y = "Truth")

ggsave("outputs/07_confusion_matrix.png", p_cm, width = 5, height = 4, dpi = 150)

message("Done. Check the outputs/ folder for figures and metrics.")