rm(list = ls())
gc()

# Load libraries

library(tidyverse)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(caret)
library(magrittr)
library(purrr)
library(xgboost)
library(Matrix)
library(MatrixModels)
library(data.table)
library(lubridate)


#---------- read in datasets ----------#
options(scipen = 9999)

# will read in the variables as characters, and then change back to factors later
train <- read.csv("/Users/jasonzivkovic/Documents/PumpItUp_DrivenDataComp/DataFiles/training_set_values.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))
train_labels <- read.csv("/Users/jasonzivkovic/Documents/PumpItUp_DrivenDataComp/DataFiles/training_set_labels.csv", na.strings = c("NA", ""))
test <- read.csv("/Users/jasonzivkovic/Documents/PumpItUp_DrivenDataComp/DataFiles/test_set_values.csv", na.strings = c("NA", ""))

# join status_group variable to rtaining set
train <- train %>%
  left_join(train_labels, by = "id")

# create status_group variable in test to be able to join the two DFs for preprocessing
# will be able to subset later to split test by calling is.na(status_group) 
test$status_group <- NA

# join the two datasets
joined <- rbind(train, test)


#---------- Inspect the data ----------#
# where are the NAs
print(colSums(is.na(joined)))

str(train)

#########################################################
#---------- Feature selection and engineering ----------#
#########################################################

##### Selection #####

# Can remove waterpoint_type_group variable - waterpoint_type shows there is a difference in the 
# association between "communal standpipe" and "communal standpipe multiple".
joined$waterpoint_type_group <- NULL

# source will be used, can drop source_type; over 75% of source = lake are non-functional, whereas source = river is only ~30%
joined$source_type <- NULL
joined$source_class <- NULL

# quantity and quantity_group are identical, will drop status_group
joined$quantity_group <- NULL

# Water quality has additional levels that appear to differ in the status of the well so I will drop quality group
joined$quality_group <- NULL

# payment and payment_type the same - can drop one.
joined$payment_type <- NULL

# will go granular and only keep 'extraction_type'
joined$extraction_type_group <- NULL
joined$extraction_type_class <- NULL

# will remove this also 
joined$permit <- NULL

# will drop scheme_name - duplicate
joined$scheme_name <- NULL

# don't need all the geographical variables so will drop some
joined$subvillage <- NULL
# joined$region <- NULL
joined$region_code <-NULL
joined$district_code <- NULL
joined$lga <- NULL
joined$ward <- NULL

# will drop permit out - no idea what this even is and schema doesn't mention anything
joined$num_private <- NULL

# these next few have too many unique entries
joined$wpt_name <- NULL
joined$installer <- NULL
joined$funder <- NULL

# only one level in recorded_by
joined$recorded_by <- NULL

# consolidated version of managemet
joined$management_group <- NULL



##### Engineering #####

# change date_recorded to a date variable
joined$date_recorded <- ymd(joined$date_recorded)

# bin construction_year and create "unknown" for NAs. Then remove construction_year
joined$construction_year_bin <- cut(joined$construction_year, seq(1960, 2020, 10)) %>% as.character()
joined$construction_year_bin[is.na(joined$construction_year_bin)] <- "Unknown"
joined$construction_year <- NULL

# I will include scheme_management in the model, but exclude scheme_name. Will also replace NAs with "unknown"
joined$scheme_management[is.na(joined$scheme_management)] <- "Unknown"

# replace NAs in public_meeting with "Unknown"
joined$public_meeting[is.na(joined$public_meeting)] <- "Unknown"

# bin population variable into population_bin, then drop population
joined$population_bin <- cut(joined$population, c(0,10,20,50,100,100000), include.lowest = TRUE) %>% as.character()
joined$population <- NULL



######## These can be added back in but will need to engineer them:
joined$amount_tsh <- NULL
joined$latitude <- NULL
joined$longitude <- NULL


#---------- Convert character variables to Factor ----------#
varlist <- function (df=NULL,type=c("numeric","factor","character"), exclude=NULL) {
  vars <- character(0)
  if (any(type %in% "numeric")) {
    vars <- c(vars,names(df)[sapply(df,is.numeric)])
  }
  if (any(type %in% "factor")) {
    vars <- c(vars,names(df)[sapply(df,is.factor)])
  }  
  if (any(type %in% "character")) {
    vars <- c(vars,names(df)[sapply(df,is.character)])
  }  
}

chars <- varlist(df = joined, type = "character")

joined %<>% mutate_at(chars, funs(factor(.)))


##################################################
#---------- Separate datasets back out ----------#
##################################################

joined_train <- joined %>%
  filter(!is.na(status_group))

joined_test <- joined %>%
  filter(is.na(status_group))

train_id <- joined_train$id
test_id <- joined_test$id
train_label <- train_labels$status_group

joined_train$id <- NULL
# joined_train$status_group <- NULL
# 
# joined_test$id <- NULL
joined_test$status_group <- NULL


# Total number of rows in the joined_train data frame
n <- nrow(train)

# Number of rows for the joined_train training set (80% of the dataset)
n_train <- round(0.80 * n) 

# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices <- sample(1:n, n_train)

# Subset the joined_train frame to training indices only
joined_train_train <- joined_train[train_indices, ]  

# Exclude the training indices to create the joined_train test set
joined_train_test <- joined_train[-train_indices, ] 
train_test_outcome <- joined_train_test$status_group

joined_train_test$status_group <- NULL

joined_train_train$id <- NULL
joined_train_test$id <- NULL


######################################
#---------- Model Training ----------#
######################################

model3 <- train(
  status_group ~ .,
  tuneLength = 1,
  data = joined_train, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)


predict_train <- predict(model3, joined_train)

confusionMatrix(train_label, predict_train)

joined_test$status_group <- predict(model3, joined_test)


submission3 <- joined_test %>% select(id, status_group)

write.csv(submission3, "submission4.csv", row.names = FALSE)

###########################################################################################################################
##### Model Score: 0.8044. Best score yet. Model trained on the full training set #####

# model3 variables: date_recorded + basin + public_meeting + scheme_management + extraction_type + management + payment +
# water_quality + quantity + source + waterpoint_type + construction_year_bin + population_bin

###########################################################################################################################


model5 <- train(
  status_group ~ .,
  tuneLength = 1,
  data = joined_train, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)


predict_train <- predict(model5, joined_train)

confusionMatrix(train_label, predict_train)

joined_test$status_group <- predict(model5, joined_test)


submission5 <- joined_test %>% select(id, status_group)

write.csv(submission5, "submission5.csv", row.names = FALSE)

###########################################################################################################################
##### Model Score: 0.8079. Best score yet. Model trained on the full training set #####

# model5 variables: date_recorded + basin + public_meeting + scheme_management + extraction_type + management + payment +
# water_quality + quantity + source + waterpoint_type + construction_year_bin + population_bin + gps_height

###########################################################################################################################


model6 <- train(
  status_group ~ .,
  tuneLength = 1,
  data = joined_train, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)


predict_train <- predict(model6, joined_train)

confusionMatrix(train_label, predict_train)

joined_test$status_group <- predict(model6, joined_test)


submission6 <- joined_test %>% select(id, status_group)

write.csv(submission6, "submission6_v2.csv", row.names = FALSE)


###########################################################################################################################
##### Model Score: 0.8091. Best score yet. Model trained on the full training set #####

# model6 variables: date_recorded + basin + public_meeting + scheme_management + extraction_type + management + payment +
# water_quality + quantity + source + waterpoint_type + construction_year_bin + population_bin + gps_height + region

###########################################################################################################################


model8 <- train(
  status_group ~ .,
  tuneLength = 1,
  data = joined_train, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)


predict_train <- predict(model8, joined_train)

confusionMatrix(train_label, predict_train)

joined_test$status_group <- predict(model8, joined_test)


submission8 <- joined_test %>% select(id, status_group)

write.csv(submission8, "submission8.csv", row.names = FALSE)


###########################################################################################################################
##### Model Score: 0.8077. Not the best score. Will look to bin amount_tsh as this model only had the raw variable #####

# model8 variables: date_recorded + basin + public_meeting + scheme_management + extraction_type + management + payment +
# water_quality + quantity + source + waterpoint_type + construction_year_bin + population_bin + gps_height + region + amount_tsh

###########################################################################################################################





