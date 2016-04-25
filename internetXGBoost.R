## Traffic donloaded from:
## https://datamarket.com/data/set/232j/internet-traffic-data-in-bits-from-a-private-isp-with-centres-in-11-european-cities-the-data-corresponds-to-a-transatlantic-link-and-was-collected-from-0657-hours-on-7-june-to-1117-hours-on-31-july-2005-hourly-data#

setwd('C:/Dropbox/Presentation/Veri Bilimi')
library(data.table)
library(bit64)
library(xgboost)
library(lubridate)
library(forecast)
library(Metrics)

## Load data
dataSet = fread('internet-traffic-data-in-bits-fr.csv', stringsAsFactors=F)

## Rename columns
setnames(dataSet, c('Time','Traffic'))

## Plot dataSet
plot(dataSet$Traffic, col='blue', pch=19, ylab='bits', type='l')

##### Date-time related features #####
dataSet[,Week:=factor(week(Time))]
dataSet[,Month:=factor(month(Time))]
dataSet[,DayOfWeek:=factor(wday(Time))]
dataSet[,DayOfMonth:=factor(mday(Time))]
dataSet[,DayOfYear:=factor(yday(Time))]
dataSet[,Hour:=factor(hour(Time))]
######################################

##### Traffic related features #####
dataSet[,Traffic:=as.numeric(Traffic)]
dataSet[,Traffic:=log(Traffic)]
## Set lag
lag = as.integer(24)
## Lag Traffic values
dataSet[,lag24:=shift(Traffic, n=lag, fill=NA)]
## Lag difference values
dataSet[,Diff:=c(NA,diff(Traffic,1,1))] # difference of target
dataSet[,lagDiff:=shift(Diff, n=lag, fill=NA)]
## Lag second difference values
dataSet[,Diff2:=c(NA,diff(Diff,1,1))] # second difference of target
dataSet[,lagDiff2:=shift(Diff2, n=lag, fill=NA)]
#################################

## Convert character variables to factor
charClass = colnames(dataSet)[sapply(dataSet, is.character)]
dataSet[,(charClass):=lapply(.SD, function(x) as.integer(factor(x))), .SDcols=charClass]

## Plot time-series and histogram
plot(dataSet$Traffic, col='blue', pch=19, ylab='bits', type='l')
hist(dataSet$Traffic, nclass='FD', col='orange')

## Create train, validation and test partitions
dataSet = dataSet[-seq(lag+2),]
t = nrow(dataSet) # length of the time series
h = 24            # horizon (forecast range, test length)
v = 2 * h         # validation length is 'n' times of validation window
train = 1:(t-v-h)
valid = (1+t-v-h):(t-h)
test  = (1+t-h):t

## Prepare data for XGBoost
dataSet = model.matrix(~.-1, dataSet) # one-hot encodingaSet = data.matrix(dataSet)
label = dataSet[,'Traffic']
remove = c('Traffic','Diff','Diff2')
features = colnames(dataSet)[!colnames(dataSet) %in% remove]
dtrain = xgb.DMatrix(dataSet[train,features], label=label[train])
dvalid = xgb.DMatrix(dataSet[valid,features], label=label[valid])
watchlist = list(val=dvalid, train=dtrain)

param = list('objective' = 'reg:linear',
             'eval_metric' = 'rmse',
             'booster' = 'gbtree',
             'max_depth' = 6,
             'eta' = 0.3,
             'colsample_bytree' = 0.8,
             'subsample' = 0.8)
nrounds = 5000
model = xgb.train(params = param,
                  data = dtrain,
                  nrounds = nrounds,
                  verbose = 1,
                  print.every.n = 40,
                  early.stop.round = 200,
                  watchlist = watchlist,
                  maximize = FALSE,
                  nthread = 4)
bestRound = model$bestInd

## Predict train, validation, and test data
predTra = predict(model, dataSet[train,features], ntreelimit=bestRound)
predVal = predict(model, dataSet[valid,features], ntreelimit=bestRound)
pred = predict(model, dataSet[test,features], ntreelimit=bestRound)

##### Summary #####
rmse(label[valid], predVal)
rmse(label[test], pred)
## Plot validation predictions
plot(valid, label[valid], xlab='hours', ylab='log(bits)')
lines(valid, predVal, col='green')
## Plot test predictions
plot(test, label[test], xlab='hours', ylab='log(bits)')
lines(test, pred, col='blue')
## Plot train, valdation, and test predictions
plot(label, xlab='hours', ylab='log(bits)', col='darkgray')
lines(train, predTra, col='red')
lines(valid, predVal, col='green')
lines(test, pred, col='blue')

## Variable importance
importanceMatrix = xgb.importance(features, model=model)
xgb.plot.importance(importanceMatrix[1:40])
###################

# ## Monthly median. Calculated excluding test set.
# mm = dataSet[1:(t-h), median(Traffic), by=.(Month)]
# setnames(mm, c('Month','MonthlyMedian'))
# dataSet = merge(dataSet, mm, by='Month', sort=FALSE)
# 
# ## Weekly median. Calculated excluding test set.
# wm = dataSet[1:(t-h), median(Traffic), by=.(Week)]
# setnames(wm, c('Week','WeeklyMedian'))
# dataSet = merge(dataSet, wm, by='Week', sort=FALSE)
# 
# ## Daily median. Calculated excluding test set.
# dm = dataSet[1:(t-h), median(Traffic), by=.(DayOfWeek)]
# setnames(dm, c('DayOfWeek','DailyMedian'))
# dataSet = merge(dataSet, dm, by='DayOfWeek', sort=FALSE)