library(caret)
library(xgboost)
library(Matrix) 

# CARE: Both packages contain predict function 
caretPredict <- caret::predict.preProcess 
xgbPredict <- xgboost::predict 


## Import and drop ID/targets
dataTrain <- read.table('Train.csv', header=TRUE, sep=',') 
dataSubmit <- read.table('Test.csv', header=TRUE, sep=',') 
Y <- dataTrain["TARGET"] 
X <- dataTrain[2:370] 
ID <- dataSubmit$ID
XSubmit <- dataSubmit[2:370] 


## Fix var3 (-999999) error code 
#table(X$var3) 
#summary(X$var3) 
#X$var3[X$var3==-999999] = 0 
#XSubmit$var3[XSubmit$var3==-999999] = 0 
#table(X$var3) summary(X$var3) 


## Fix 9999999999 error code 
#XRep = matrix(nrow=1, ncol=dim(X)[2]) 
#for (m in seq(1, dim(X)[2])) 
#{ 
#  print(m) 
#  rIdx = X[,m] == 9999999999 
#  X[rIdx,m] = 0 
#  XRep[1,m] = sum(rIdx) 
#  rIdx = XSubmit[,m] == 9999999999 
#  XSubmit[rIdx, m] = 0 
#} 

countX <- function(x, n) { sum(x==n) }

## Add zero count 
X$ZeroCount = apply(X, 1, countX, n=0) 
XSubmit$ZeroCount = apply(XSubmit, 1, countX, n=0) 
plot(X$ZeroCount) 

## Add 9999999999 count
X$e9sCount = apply(X, 1, countX, n=9999999999) 
XSubmit$e9sCount = apply(XSubmit, 1, countX, n=9999999999) 
plot(X$e9sCount) 

## var38 is either net value or customer value
## var4 is number of bank products
## Do any customers have a value   

## Get age (var15) and other happy vairables
SMV5H2 = XSubmit['saldo_medio_var5_hace2']
SV33 = XSubmit['saldo_var33']
var38 = XSubmit['var38']
V21 = XSubmit['var21']
# XAge = X$var15 
XSubmitAge = XSubmit$var15 

nv = XSubmit['num_var33']+XSubmit['saldo_medio_var33_ult3']+XSubmit['saldo_medio_var44_hace2']+XSubmit['saldo_medio_var44_hace3']+
  XSubmit['saldo_medio_var33_ult1']+XSubmit['saldo_medio_var44_ult1']


## Remove zero var predictors
toRemove <- c()
count = 0
for (c in names(X))
{
  if (sd(X[[c]]) == 0)
  {
    count = count+1
    cat("\n", c, "is constant")
    X[c] <- NULL
    XSubmit[c] <- NULL
  }
}


## Remove identical features
features_pair <- combn(names(X), 2, simplify = F) 
toRemove <- c() 
for(pair in features_pair) 
{
  f1 <- pair[1] 
  f2 <- pair[2] 
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) 
  { 
    if (all(X[[f1]] == X[[f2]])) 
    { cat(f1, "and", f2, "are equals.\n") 
      toRemove <- c(toRemove, f2) 
    } 
  } 
} 

for (c in toRemove)
{
  X[c] <- NULL
  XSubmit[c] <- NULL
}


## Before logging var38
#X$var38Yes <- as.numeric(X$var38 == 117310.979016494) # Yes = no value = "error code"
#XSubmit$var38Yes <- as.numeric(XSubmit$var38 == 117310.979016494) 

## Log var 38 plot(X$var38) 
X$var38 <- log(X$var38) 
XSubmit$var38 <- log(XSubmit$var38) 
plot(X$var38) 
plot(XSubmit$var38)

## Remove linear dependencies 
#linCombos <- findLinearCombos(X) 
# rm = linCombos$remove 
#if (!is.null(rm)) 
#{ 
#  X = X[, -rm] 
#  XSubmit = XSubmit[, -rm] 
#} 


## Add some more features
#Calucalte value per product
#X$VPP = X$num_var4/X$var38
#XSubmit$VPP = XSubmit$num_var4/XSubmit$var38
#hist(X$VPP)
#hist(XSubmit$VPP)

# Greater prop are Y==1 when they have no products and have no value
#X$NoPNoV = as.numeric(X$num_var4==0 & X$var38Yes==1)
#XSubmit$NoPNoV = as.numeric(XSubmit$num_var4==0 & XSubmit$var38Yes==1)
#hist(X$NoPNoV [Y==1])
#hist(X$NoPNoV [Y==0])
#sum(X$NoPNoV [Y==1])/length(X$NoPNoV [Y==1])
#sum(X$NoPNoV [Y==0])/length(X$NoPNoV [Y==0])
#hist(X$NoPNoV)
#hist(XSubmit$NoPNoV)

# Greater prop are Y==1 when they have no products but seem to have a value
#X$NoPButHasV = as.numeric(X$num_var4==0 & X$var38Yes==0)
#XSubmit$NoPButHasV = as.numeric(XSubmit$num_var4==0 & XSubmit$var38Yes==0)
#hist(X$NoPButHasV[Y==1])
#hist(X$NoPButHasV[Y==0])
#sum(X$NoPButHasV[Y==1])/length(X$NoPButHasV[Y==1])
#sum(X$NoPButHasV[Y==0])/length(X$NoPButHasV[Y==0])
#hist(X$NoPButHasV)
#hist(XSubmit$NoPButHasV)

#X$logAbsv30 = log(abs(X$saldo_var30))
#X$logAbsv30[is.na(X$logAbsv30)] <- -1
#X$logAbsv30[is.infinite(X$logAbsv30)] <- 0
#XSubmit$logAbsv30 = log(abs(XSubmit$saldo_var30))
#XSubmit$logAbsv30[is.na(XSubmit$logAbsv30)] <- -1
#XSubmit$logAbsv30[is.infinite(XSubmit$logAbsv30)] <- 0
#hist(X$logAbsv30[Y==1])
#hist(X$logAbsv30[Y==0])
#sum(X$logAbsv30[Y==1])/length(X$logAbsv30[Y==1])
#sum(X$logAbsv30[Y==0])/length(X$logAbsv30[Y==0])
#hist(XSubmit$logAbsv30)

#X$LAV30v15 = X$logAbsv30/X$var15
#XSubmit$LAV30v15 = XSubmit$logAbsv30/XSubmit$var15
#hist(X$LAV30v15[Y==1])
#hist(X$LAV30v15[Y==0])
#sum(X$LAV30v15[Y==1])/length(X$LAV30v15[Y==1])
#sum(X$LAV30v15[Y==0])/length(X$LAV30v15[Y==0])
#hist(XSubmit$LAV30v15)

##---limit vars in test based on min and max vals of train 
# From https://www.kaggle.com/lucapolverini/ 
# santander-customer-satisfaction/under-23-year-olds-are-always-happy/run/216651 
# Do this because XGB is bad at extrapolation
print('Setting min-max lims on test data') 
for(f in colnames(X))
{ 
  lim <- min(X[,f]) 
  XSubmit[XSubmit[,f]<lim,f] <- lim 
  
  lim <- max(X[,f]) 
  XSubmit[XSubmit[,f]>lim,f] <- lim 
} 


## Advanced train model 

YTrain = Y[,1] # should be integer, not data.frame 
dtrain <- xgb.DMatrix(data = data.matrix(X),
                      label=YTrain) 
watchlist <- list(train=dtrain) 

bst <- xgb.train(data=dtrain, 
                 max.depth=5, 
                 eta=0.0202000, 
                 nthread = 6, 
                 nround=560, 
                 watchlist=watchlist, 
                 objective = "binary:logistic", 
                 booster = "gbtree", 
                 eval.metric = "error", 
                 eval.metric = "logloss",
                 eval_metric = "auc", 
                 subsample = 0.6815, 
                 colsample_bytree=0.701) 

importance_matrix <- xgb.importance(model = bst, feature_names = names(X)) 
print(importance_matrix) 
plot(importance_matrix$Feature, importance_matrix$Gain)
plot(importance_matrix$Feature, importance_matrix$Frequence)

cbind(importance_matrix$Feature, importance_matrix$Gain)

## Run model 
yPred <- xgbPredict(bst, data.matrix(XSubmit)) 
plot(yPred) 

# Set happy groups
yPred[XSubmitAge<23] = 0 
yPred[SMV5H2>160000] = 0
yPred[SV33>0] = 0
yPred[var38 > 3988596] = 0
yPred[V21>7500]=0
yPred[nv > 0] = 0

plot(yPred)
hist(yPred, breaks=200)

# yPred[yPred<0.0075]=0
idx = yPred>0.15 &  yPred<0.25
hist(yPred[idx], breaks = 200) 
sum(yPred)
yPred[idx] = yPred[idx] * 0.9
hist(yPred[idx], breaks = 200) 
sum(yPred)
hist(yPred, breaks=200)

sumYPred = c()
bias = c()
for (b in seq(1,1000))
{
  bias[b] = 0+b/1000
  tmp = yPred
  tmp[yPred<bias[b]] = 0
  sumYPred[b] = sum(tmp)
  print(bias[b])
  print(sumYPred[b])
}
plot(bias, sumYPred)

submission <- data.frame(ID=ID, TARGET=yPred) 
cat("saving the submission file\n") 
write.csv(submission, "submission.csv", row.names = F)