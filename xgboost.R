
library(tidyverse) # metapackage with lots of helpful functions

## Running code

# In a notebook, you can run a single code cell by clicking in the cell and then hitting 
# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 
# you can run code by highlighting the code you want to run and then clicking the blue arrow
# at the bottom of this window.

## Reading in files

# You can access files from datasets you've added to this kernel in the "../input/" directory.
# You can see the files added to this kernel by running the code below. 

#list.files(path = "../input/sf-crime")

## Saving data

# If you save any files or images, these will be put in the "output" directory. You 
# can see the output directory by committing and running your kernel (using the 
# Commit & Run button) and then checking out the compiled version of your kernel.

library(reshape2)
library(data.table)
library(caret)
library(dplyr)
library(lubridate)
library(gbm)
library(randomForest)
library(tree)
library(rpart)
library(xgboost)
set.seed(825)

train <- read.csv(unzip(zipfile = "../input/sf-crime/train.csv.zip"))
train <- train[complete.cases(train),]
test <- read.csv(unzip(zipfile = "../input/sf-crime/test.csv.zip"))

train$Dates <- ymd_hms(train$Dates,tz=Sys.timezone())
train$Year <- year(train$Dates)
train$DayOfWeek <- factor(weekdays(train$Dates))
train$Category <- factor(train$Category)
train$month <- month(train$Dates)
train$DayOfMonth <- day(train$Dates)
train$Hour <- hour(train$Dates)
train.x <- train[,c(1,4,5,8,9)]
train.x$Dates <- ymd_hms(train.x$Dates,tz=Sys.timezone())
train$Year <- year(train$Dates)
train.x$Month <- month(train.x$Dates)
train.x$HourofDay<- hour(train.x$Dates)
train.x$DayofMonth <- day(train.x$Dates)
train.x$Year <- year(train.x$Dates)
train.x$PdDistrict <- factor(train.x$PdDistrict)
train.x <- train.x[,-1]



train.x <- train.x[complete.cases(train.x), ]

idx <- with(train.x, which(Y == 90))
transform <- preProcess(train.x[-idx, c('X', 'Y')], method = c('center', 'scale', 'pca'))
pc <- predict(transform, train.x[, c('X', 'Y')]) 
train.x$X <- pc$PC1
train.x$Y <- pc$PC2

#pp_df <- preProcess(train.x,
#                     method = c("center", "scale", "YeoJohnson"))

#transformed <- predict(pp_df, newdata = train.x) #878049
transformed <- train.x[,-2]
#making sure test data has same columns 
keep<- colnames(transformed)
test$Dates <- ymd_hms(test$Dates,tz=Sys.timezone())
test$Month <- month(test$Dates)
test$HourofDay<- hour(test$Dates)
test$DayofMonth <- day(test$Dates)
test$DayOfWeek <-factor(test$DayOfWeek)
test$PdDistrict <- factor(test$PdDistrict)
test$Year <- year(test$Dates)
test.2 <- test[,keep] #581 x 128
pc.t <- predict(transform, newdata =test.2)
test.2$X <- pc.t$PC1
test.2$Y <- pc.t$PC2

transformed_test <- cbind.data.frame(Id = test$Id,test.2)
transformed_w_category <- cbind.data.frame(Category = factor(train$Category),transformed)


fitControl <- trainControl(## 10-fold CV
                           method = "cv",
                           number = 5)






#idx <- which(!is.na(transformed_w_category$Category))
classes <- sort(unique(transformed_w_category$Category))
m <- length(classes)
transformed_w_category$Class <- as.integer(factor(transformed_w_category$Category, levels=classes)) - 1
#dim(transformed_w_category) #878049 x 9

feature.names <- names(transformed_w_category)[which(!(names(transformed_w_category) %in% c('Id', 'Address', 'Dates', 'Category', 'Class')))]
for (feature in feature.names){
    if (class(transformed_w_category[[feature]]) == 'character'){
        cat(feature, 'converted\n')
        levels <- unique(transformed_w_category[[feature]])
        transformed_w_category[[feature]] <- as.integer(factor(transformed_w_category[[feature]], levels=levels))
    }
}

param <- list(
                #nthread             = 4,
                booster             = 'gbtree',
                objective           = 'multi:softprob',
                num_class           = m,
                eta                 = .95,
                #gamma               = 0,
                max_depth           = 6,
                #min_child_weigth    = 1,
                max_delta_step      = 1
                #subsample           = 1,
                
)

h <- sample(1:length(transformed_w_category$Category), floor(9*length(transformed_w_category$Category)/10))
dval <- xgb.DMatrix(data=data.matrix(transformed_w_category[-h, feature.names]), label=transformed_w_category$Class[-h])
dtrain <- xgb.DMatrix(data=data.matrix(transformed_w_category[h, feature.names]), label=transformed_w_category$Class[h])
watchlist <- list(val=dval, train=dtrain)
bst <- xgb.train( params            = param,
                  data              = dtrain,
                  watchlist         = watchlist,
                  verbose           = 1,
                  eval_metric       = 'mlogloss',
                  nrounds           = 14
)


dtest <- xgb.DMatrix(data=data.matrix(transformed_test[,feature.names]))
prediction <- predict(bst, dtest)
prediction <- sprintf('%f', prediction)
prediction <- cbind(transformed_test$Id, t(matrix(prediction, nrow=m)))
dim(prediction)

colnames(prediction) <- c('Id', as.vector(classes))

write.csv(prediction, 'submission.csv', row.names=FALSE, quote=FALSE)

