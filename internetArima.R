## Traffic donloaded from:
## https://datamarket.com/data/set/232j/internet-traffic-data-in-bits-from-a-private-isp-with-centres-in-11-european-cities-the-data-corresponds-to-a-transatlantic-link-and-was-collected-from-0657-hours-on-7-june-to-1117-hours-on-31-july-2005-hourly-data#

setwd('C:/Dropbox/Presentation/Veri Bilimi')
library(data.table)
library(bit64)
library(lubridate)
library(forecast)
library(tsDyn)
library(Metrics)

## Load data
dataSet = fread('internet-traffic-data-in-bits-fr.csv', stringsAsFactors=F)

## Rename columns
setnames(dataSet, c('Time','Traffic'))

## Convert dataSet to time series
dataSet[,Traffic:=as.numeric(Traffic)]
dataSet[,Traffic:=log(Traffic)]
tSerie = ts(dataSet$Traffic, start=1, frequency=24)

## Prepare train and test partitions
t = length(tSerie) # length of time series
h = 24 # forecast range (horizon) - test range
N = 6 # number of total forecast
train = 1:(t-h)
test = (t-h+1):t

## ARIMA model
fit = auto.arima(tSerie[train], lambda=NULL); fit
Acf(tSerie[train], main='')
Pacf(tSerie[train], main='')
fc = forecast(fit, h=h)
plot(fc, include=80)
lines(test, tSerie[test], col='red')
# RMSE score
rmse(tSerie[test], fc$mean)

R = vector(mode='numeric')
for (i in 1:N){
  train = 1:(t-h-N+i)
  test = (t-h-N+i+1):(t-N+i)
  fit = auto.arima(tSerie[train], lambda=NULL)
  fc = forecast(fit, h=h)
  plot(fc, include=80)
  lines(test, tSerie[test], col='red')
  R[i] = rmse(tSerie[test], fc$mean)
}
mean(R)
sd(R)

## Recurrence plot
# In descriptive statistics and chaos theory,
# a recurrence plot (RP) is a plot showing, for a given moment in time,
# the times at which a phase space trajectory visits roughly
# the same area in the phase space.
library(tseriesChaos)
recurr(tSerie, m=2, d=1, levels=c(0,0.2,1))
recurr(tSerie, m=2, d=1)
recurr(tSerie, m=3, d=2)
