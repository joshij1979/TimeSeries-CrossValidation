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
## Lag target values
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

## Create train, validation, and test partition values
dataSet = dataSet[-seq(lag+2),]
t = nrow(dataSet) # length of the time series
h = 24            # horizon (forecast range, test length)
w = 2 * h         # validation window
M = 4 * w         # validation length
N = 6 * 1         # number of forecasts
V = M - w         # rolling validation length

## Prepare data for XGBoost
dataSet = model.matrix(~.-1, dataSet) # one-hot encoding
label = dataSet[,'Traffic']
remove = c('Time','Traffic','Diff','Diff2')
features = colnames(dataSet)[!colnames(dataSet) %in% remove]

param = list('objective' = 'reg:linear',
             'eval_metric' = 'rmse',
             'booster' = 'gbtree',
             'max_depth' = 6,
             'eta' = 0.3,
             'colsample_bytree' = 0.8,
             'subsample' = 0.8)

meanRmse = vector(mode='numeric')
for (i in 1:N){
  test = (t-N-h+1+i):(t-N+i)
  pred = data.frame(matrix(vector(), h, V))
  for (j in 1:V){
    train = 1:(t-N-M-h+i+j)
    valid = (t-N-M-h+1+i+j):(t-N-M-h+w+i+j)
    dtrain = xgb.DMatrix(dataSet[train,features], label=label[train])
    dvalid = xgb.DMatrix(dataSet[valid,features], label=label[valid])
    watchlist = list(val=dvalid, train=dtrain)
    nrounds = 5000
    model = xgb.train(params = param,
                      data = dtrain,
                      nrounds = nrounds,
                      verbose = 1,
                      print.every.n = 100,
                      early.stop.round = 200,
                      watchlist = watchlist,
                      maximize = FALSE,
                      nthread = 4)
    ## Predict test data
    pred[,j] = predict(model, dataSet[test,features], ntreelimit=model$bestInd)
  }
  ## Mean prediction and RMSE of models
  meanPred = rowMeans(pred)
  meanRmse[i] = rmse(label[test], meanPred)
  ## Plot predictions
  plot(label[test], col='darkgreen', pch=19, xlab='hours', ylab='log(bits)')
  lines(label[test], col='darkgreen', lty=2)
  points(meanPred, col='orange', pch=19)
  lines(meanPred, col='orange', lty=2)
}

## Summary
mean(meanRmse)
sd(meanRmse)
