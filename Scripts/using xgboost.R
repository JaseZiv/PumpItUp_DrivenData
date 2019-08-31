rm(list = ls())
gc()

# Load libraries

library(tidyverse)
library(caret)
library(magrittr)
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

#Create new column in test 
test$status_group <- 0

#Subset label so that it only contains the label (target variable)
label <- subset(train_labels, select = status_group )

#Combine the train and the label data sets
train<-cbind(train,label)

#Create new status_group column so test and train and same number of columns. 
#Required when building the model
train$status_group<-0

#Designate columns as train and test
train$tst <- 0
test$tst <- 1

#Combine train and test into one dataset
joined <- rbind(train,test)



y <- train_labels$status_group

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

# # bin construction_year and create "unknown" for NAs. Then remove construction_year
# joined$construction_year_bin <- cut(joined$construction_year, seq(1960, 2020, 10)) %>% as.character()
# joined$construction_year_bin[is.na(joined$construction_year_bin)] <- "Unknown"
# joined$construction_year <- NULL

# I will include scheme_management in the model, but exclude scheme_name. Will also replace NAs with "unknown"
joined$scheme_management[is.na(joined$scheme_management)] <- "Unknown"

# replace NAs in public_meeting with "Unknown"
joined$public_meeting[is.na(joined$public_meeting)] <- "Unknown"

# # bin population variable into population_bin, then drop population
# joined$population_bin <- cut(joined$population, c(0,10,20,50,100,100000), include.lowest = TRUE) %>% as.character()
# joined$population <- NULL



######## These can be added back in but will need to engineer them:
# joined$amount_tsh <- NULL
# joined$latitude <- NULL
# joined$longitude <- NULL


#Separate data into train and test set
data_train <- joined[joined$tst==0,]
data_test <- joined[joined$tst==1,]

#Create test set that doesn't contain the ID column. I did this because the test and train
#datsets need to have the same number of columns when making predictions.
data_test.noID<-subset(data_test, select = -id)

#Remove the id and status group columns from the train dataset. I don't want these columns
#to affect the the model
data_train<- data_train %>% select(-id, -status_group)

#Convert data frames to numeric matrices. Xgboost requires user to enter data as a numeric matrix
data_test.noID <- data_test.noID %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer())) %>%
  data.matrix()

data_train <- data_train %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer())) %>%
  data.matrix()

label<-as.numeric(train_labels$status_group)

#Create a xgb.DMatrix which is the best format to use to create an xgboost model
train.DMatrix <- xgb.DMatrix(data = data_train,label = label, missing = NA)


#For loop to run model 11 time with different random seeds. Using an ensemble technique such as this
#improved the model performance

#Set i=2 because the first column is for the id variable
i=2

#Create data frame to hold the 11 solutions developed by the model
solution.table<-data.frame(id=data_test[,"id"])
for (i in 2:12){
  #Set seed so that the results are reproducible
  set.seed(i)
  
#Cross validation to determine the number of iterations to run the model.
#I tested this model with a variety of parameters to find the most accurate model
  xgb.tab = xgb.cv(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                 nrounds = 800, nfold = 4, early_stopping_rounds = 500, num_class = 4, maximize = FALSE,
                 evaluation = "merror", eta = .01, max_depth = 8, colsample_bytree = .8, print_every_n = 100)

#Create variable that identifies the optimal number of iterations for the model
  min.error.idx = which.min(xgb.tab$evaluation_log$test_merror_mean)


#Create model using the same parameters used in xgb.cv
  model <- xgboost(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                 eval_metric = "merror", nrounds = min.error.idx,
                 num_class = 4,eta = .01, max_depth = 8, colsample_bytree = .8, print_every_n = 100)



#Predict. Used the data_test.noID because it contained the same number of columns as the train.DMatrix
#used to build the model.
  predict <- predict(model,data_test.noID)

#Modify prediction labels to match submission format
  predict[predict==1]<-"functional"
  predict[predict==2]<-"functional needs repair"
  predict[predict==3]<-"non functional"

#View prediction
  table(predict)

#Add the solution to column i of the solutions data frame. This creates a data frame with a column for
#each prediction set. Each prediction is a vote for that prediction. Next I will count the number of votes
#for each prediction as use the element with the most votes as my final solution.
  solution.table[,i]<-predict
  }

#Count the number of votes for each solution for each row
solution.table.count<-apply(solution.table,MARGIN=1,table)

#Create a vector to hold the final solution
predict.combined<-vector()

x=1
#Finds the element that has the most votes for each prediction row
for (x in 1:nrow(data_test)){
  predict.combined[x]<-names(which.max(solution.table.count[[x]]))}

#View the number of predictions for each classification
table(predict.combined)

#Create solution data frame
solution<- data.frame(id=data_test[,"id"], status_group=predict.combined)

#View the first five rows of the solution to ensure that it follows submission format rules
head(solution)

#Create csv submission file
write.csv(solution, file = "Water_solution - xgboost 45.csv", row.names = FALSE)

#Calculate the importance of each variable to the model.
#Used this function to remove variables from the model variables which don't contribute to the model.
importance <- xgb.importance(feature_names = colnames(data_train), model =model)
importance
xgb.plot.importance(importance_matrix = importance)
# 
# #score .8247
# 

# 
# solution <- data.frame(id=data_test[,"id"], status_group=predict)
# 
# head(solution)
# 
# write.csv(solution, file = "Water_solution - xgboost_v1.csv", row.names = FALSE)
# 

