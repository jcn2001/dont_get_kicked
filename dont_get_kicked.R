library(tidyverse)
library(tidymodels) 
library(vroom)
library(glmnet)
library(embed)
library(discrim)
library(ranger)
library(bonsai)
library(lightgbm)

kicked_train <- vroom("C:/Users/Josh/Documents/stat348/dont_get_kicked/training.csv")
kicked_test <- vroom("C:/Users/Josh/Documents/stat348/dont_get_kicked/test.csv")


## set up the recipe
kicked_recipe <- recipe(IsBadBuy~.,data= kicked_train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_mutate(VNZIP1=as.factor(VNZIP1)) %>%
  step_mutate(IsBadBuy = as.factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(VehYear=as.factor(VehYear)) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate(BYRNO=as.factor(BYRNO)) %>%
  step_rm(contains('MMR')) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_other(all_nominal_predictors(), threshold = .0001) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  ## step_date(PurchDate, features=c("dow","year"))
  step_rm(PurchDate, RefId, WheelType, PRIMEUNIT, AUCGUART)

prepped_recipe <- prep(kicked_recipe)
bake(prepped_recipe, new_data=kicked_train)
bake(prepped_recipe, new_data=kicked_test)

## random forest for a binary response
rf_model_kicked <- rand_forest(mtry = tune(),
                                min_n=tune(),
                                trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# set workflow
rf_wf_kicked <- workflow() %>%
  add_recipe(kicked_recipe) %>%
  add_model(rf_model_kicked)

# grid of values to tune over
grid_rf_kicked <- grid_regular(mtry(range=c(1,10)),
                                    min_n(),
                                    levels = 5)
# split data for CV
rf_folds_kicked <- vfold_cv(kicked_train, v = 10, repeats=1)

# Run the CV
rf_CV_results_kicked <- rf_wf_kicked %>%
  tune_grid(resamples=rf_folds_kicked,
            grid=grid_rf_kicked,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
best_rf_kicked <- rf_CV_results_kicked %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_rf_wf_kicked <- rf_wf_kicked %>%
  finalize_workflow(best_rf_kicked) %>%
  fit(data=kicked_train)

## make predictions
kicked_test$MMRCurrentAuctionAveragePrice <- as.character(kicked_test$MMRCurrentAuctionAveragePrice)
kicked_test$MMRCurrentAuctionCleanPrice <- as.character(kicked_test$MMRCurrentAuctionCleanPrice)
kicked_test$MMRCurrentRetailAveragePrice <- as.character(kicked_test$MMRCurrentRetailAveragePrice)
kicked_test$MMRCurrentRetailCleanPrice <- as.character(kicked_test$MMRCurrentRetailCleanPrice)

rf_preds <- predict(final_rf_wf_kicked, new_data = kicked_test, type = "prob")

# create the file to submit to kaggle
rf_submission <- rf_preds %>%
  bind_cols(.,kicked_test) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)

vroom_write(x=rf_Preds, file ="C:/Users/Josh/Documents/stat348/dont_get_kicked/nb.csv", delim=",")

# nb model 
nb_model_kicked <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf_kicked <- workflow() %>%
  add_recipe(kicked_recipe) %>%
  add_model(nb_model_kicked)

# tune smoothness and laplace
# tuning grid
nb_tuning_grid_kicked <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)
#folds
nb_folds_kicked <- vfold_cv(kicked_train, v = 10, repeats= 1)

# cross-validation
nb_CV_results_kicked <- nb_wf_kicked %>%
  tune_grid(resamples=nb_folds_kicked,
            grid=nb_tuning_grid_kicked,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_nb_tune_kicked <- nb_CV_results_kicked %>%
  select_best(metric = "roc_auc")

# # Finalize the workflow and fit it
final_nb_wf_kicked <- nb_wf_kicked %>%
  finalize_workflow(best_nb_tune_kicked) %>%
  fit(data=kicked_train)

# make predictions with the model
kicked_test$MMRCurrentAuctionAveragePrice <- as.character(kicked_test$MMRCurrentAuctionAveragePrice)
kicked_test$MMRCurrentAuctionCleanPrice <- as.character(kicked_test$MMRCurrentAuctionCleanPrice)
kicked_test$MMRCurrentRetailAveragePrice <- as.character(kicked_test$MMRCurrentRetailAveragePrice)
kicked_test$MMRCurrentRetailCleanPrice <- as.character(kicked_test$MMRCurrentRetailCleanPrice)

nb_preds_kicked <- predict(final_nb_wf_kicked, new_data = kicked_test, type = "prob")

nb_submission <- nb_preds_kicked %>%
  bind_cols(.,kicked_test) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)

vroom_write(x=nb_submission, file ="C:/Users/Josh/Documents/stat348/dont_get_kicked/nb.csv", delim=",")



## fit a boosted model
# assign the model
boost_model_kicked <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# set up the workflow
boost_wf_kicked <- workflow() %>%
  add_recipe(kicked_recipe) %>%
  add_model(boost_model_kicked)

# tune smoothness and laplace
# tuning grid
boost_tuning_grid_kicked <- grid_regular(tree_depth(),
                                  trees(),
                                  learn_rate(),
                                  levels = 5)
#folds
boost_folds_kicked <- vfold_cv(kicked_train, v = 10, repeats= 1)

# cross-validation
boost_CV_results_kicked <- boost_wf_kicked %>%
  tune_grid(resamples=boost_folds_kicked,
            grid=boost_tuning_grid_kicked,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_boost_tune_kicked <- boost_CV_results_kicked %>%
  select_best(metric = "roc_auc")

## Finalize the workflow and fit it
final_boost_wf_kicked <- boost_wf_kicked %>%
  finalize_workflow(best_boost_tune_kicked) %>%
  fit(kicked_train)

#predictions
kicked_test$MMRCurrentAuctionAveragePrice <- as.character(kicked_test$MMRCurrentAuctionAveragePrice)
kicked_test$MMRCurrentAuctionCleanPrice <- as.character(kicked_test$MMRCurrentAuctionCleanPrice)
kicked_test$MMRCurrentRetailAveragePrice <- as.character(kicked_test$MMRCurrentRetailAveragePrice)
kicked_test$MMRCurrentRetailCleanPrice <- as.character(kicked_test$MMRCurrentRetailCleanPrice)

boost_predictions_kicked <- predict(final_boost_wf_kicked,
                             new_data=kicked_test,
                             type="prob")

# make the file to submit
boost_submission_kicked <- boost_predictions_kicked %>%
  bind_cols(.,kicked_test) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)

vroom_write(x=boost_submission_kicked, file ="C:/Users/Josh/Documents/stat348/dont_get_kicked/boost_preds.csv", delim=",")


## let's try logistic regression
log_kicked_recipe <- recipe(IsBadBuy~.,data= kicked_train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_mutate(VNZIP1=as.factor(VNZIP1)) %>%
  step_mutate(VehYear=as.factor(VehYear)) %>%
  step_mutate(BYRNO=as.factor(BYRNO)) %>%
  step_impute_mean(all_numeric_predictors()) %>%  # Impute missing values in numeric predictors
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  ## step_date(PurchDate, features=c("dow","year")) %>%
  step_rm(PurchDate, RefId, WheelType, PRIMEUNIT, AUCGUART)

log_kicked <- logistic_reg() %>%
  set_engine("glm")

log_wf_kicked <- workflow() %>%
  add_recipe(log_kicked_recipe) %>%
  add_model(log_kicked) %>%
  fit(data=kicked_train)

log_preds_kicked <- predict(log_wf_kicked,
                            new_data=kicked_test,
                            type="prob")


log_submission <- log_preds_kicked %>%
  bind_cols(.,kicked_test) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)

vroom_write(x=log_submission, file ="C:/Users/Josh/Documents/stat348/dont_get_kicked/LogisticPreds.csv", delim=",")
