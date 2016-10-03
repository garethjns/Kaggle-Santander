library(caret) 
library(xgboost) 
library(Matrix) 
library(stats)

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


countX <- function(x, n) { sum(x==n) } 
## Add zero count 
X$ZeroCount = apply(X, 1, countX, n=0) 
XSubmit$ZeroCount = apply(XSubmit, 1, countX, n=0) 
plot(X$ZeroCount) 

## Add 9999999999 count 
X$e9sCount = apply(X, 1, countX, n=9999999999) 
XSubmit$e9sCount = apply(XSubmit, 1, countX, n=9999999999) 
plot(X$e9sCount) 

## Get age (var15) and other happy vairables 
SMV5H2 = XSubmit['saldo_medio_var5_hace2'] 
SV33 = XSubmit['saldo_var33'] 
var38 = XSubmit['var38'] 
V21 = XSubmit['var21'] 
# XAge = X$var15 
XSubmitAge = XSubmit$var15 

## Fix var3 (-999999) error code 
table(X$var3) 
summary(X$var3) 
X$var3[X$var3==-999999] = 2 
XSubmit$var3[XSubmit$var3==-999999] = 2 
table(X$var3) summary(X$var3) 

## Fix 9999999999 error code 
XRep = matrix(nrow=1, ncol=dim(X)[2]) 
for (m in seq(1, dim(X)[2])) 
{ 
   print(m) 
   rIdx = X[,m] == 9999999999 
   X[rIdx,m] = 0 
   XRep[1,m] = sum(rIdx) 
   rIdx = XSubmit[,m] == 9999999999 
   XSubmit[rIdx, m] = 0 
  } 
  

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
      {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
      }
    }
  }
for (c in toRemove) 
  {
  X[c] <- NULL 
  XSubmit[c] <- NULL 
  }

## Log var 38 
plot(X$var38) 
X$var38 <- log(X$var38) 
XSubmit$var38 <- log(XSubmit$var38) 
plot(X$var38) 

## PCA
trans = preProcess(X[,1:4], 
                   method=c("BoxCox", "center", 
                            "scale", "pca"))
PC = predict(trans, X[,1:4])
X = cbind(X, PC)

trans = preProcess(XSubmit[,1:4], 
                   method=c("BoxCox", "center", 
                            "scale", "pca"))
PC = predict(trans, XSubmit[,1:4])
XSubmit = cbind(XSubmit, PC)

##---limit vars in test based on min and max vals of train 
# From https://www.kaggle.com/lucapolverini/santander-customer-satisfaction/under-23-year-olds-are-always-happy/run/216651 
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
dtrain <- xgb.DMatrix(data = data.matrix(X), label=YTrain) 
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
                 eval_metric="auc", 
                 subsample = 0.6815, 
                 colsample_bytree=0.701) 

importance_matrix <- xgb.importance(model = bst, feature_names = names(X)) 
print(importance_matrix) 
plot(importance_matrix$Feature, importance_matrix$Gain) 
plot(importance_matrix$Feature, importance_matrix$Frequence) 

## Run model 
yPred <- xgbPredict(bst, data.matrix(XSubmit))
plot(yPred) # Set happy groups 

nv = XSubmit['num_var33']+XSubmit['saldo_medio_var33_ult3']+XSubmit['saldo_medio_var44_hace2']+XSubmit['saldo_medio_var44_hace3']+
  XSubmit['saldo_medio_var33_ult1']+XSubmit['saldo_medio_var44_ult1']

yPred[XSubmitAge<23] = 0 
yPred[SMV5H2>160000] = 0 
yPred[SV33>0] = 0 
yPred[var38 > 3988596] = 0 
yPred[V21>7500]=0 
plot(yPred) 
hist(yPred)

yPred[nv > 0] = 0
yPred[XSubmit['var15'] < 23] = 0
yPred[XSubmit['saldo_medio_var5_hace2'] > 160000] = 0
yPred[XSubmit['saldo_var33'] > 0] = 0
yPred[XSubmit['var38'] > 3988596] = 0
yPred[XSubmit['var21'] > 7500] = 0
yPred[XSubmit['num_var30'] > 9] = 0
yPred[XSubmit['num_var13_0'] > 6] = 0
yPred[XSubmit['num_var33_0'] > 0] = 0
yPred[XSubmit['imp_ent_var16_ult1'] > 51003] = 0
yPred[XSubmit['imp_op_var39_comer_ult3'] > 13184] = 0
yPred[XSubmit['saldo_medio_var5_ult3'] > 108251] = 0
yPred[(XSubmit['var15']+XSubmit['num_var45_hace3']+XSubmit['num_var45_ult3']+XSubmit['var36']) <= 24] = 0
yPred[XSubmit['saldo_var5'] > 137615] = 0
yPred[XSubmit['saldo_var8'] > 60099] = 0
yPred[(XSubmit['var15']+XSubmit['num_var45_hace3']+XSubmit['num_var45_ult3']+XSubmit['var36']) <= 24] = 0
yPred[XSubmit['saldo_var14'] > 19053.78] = 0
yPred[XSubmit['saldo_var17'] > 288188.97] = 0
yPred[XSubmit['saldo_var26'] > 10381.29] = 0
yPred[XSubmit['num_var13_largo_0'] > 3] = 0
yPred[XSubmit['imp_op_var40_comer_ult1'] > 3639.87] = 0
yPred[XSubmit['saldo_medio_var13_largo_ult1'] > 0] = 0
yPred[XSubmit['num_meses_var13_largo_ult3'] > 0] = 0
yPred[XSubmit['num_var20_0'] > 0] = 0  
yPred[XSubmit['saldo_var13_largo'] > 150000] = 0

hist(yPred) 
submission <- data.frame(ID=ID, TARGET=yPred)
cat("saving the submission file\n") 
write.csv(submission, "submission.csv", row.names = F)