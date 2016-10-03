
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import auc, roc_curve
import random

# Function to score (and plot) models
def scoreMod(clf, XTest, YTest):
    yPred = clf.predict_proba(XTest) 
    fpr, tpr, _ = roc_curve(YTest, yPred[:, 1])
    plt.plot(fpr,tpr)
    score = auc(fpr, tpr)
    print(score)                          
    return yPred, score

#Training functions
nJobs = 7;
def trainPredExT(XTrain, YTrain, XTest, YTest, nEst, mD):
    # Create    
    clf = ensemble.ExtraTreesClassifier(n_estimators=nEst,
                                        max_features='auto',
                                        criterion= 'entropy',
                                        min_samples_split= 40,
                                        max_depth= mD,
                                        min_samples_leaf= 2, 
                                        n_jobs = nJobs, 
                                        random_state=rnd,
                                        verbose=1)
    # Fit                                    
    clf.fit(XTrain, YTrain) 
    # Score   
    yPred, score = scoreMod(clf, XTest, YTest)
    
    return clf, yPred, score
    

def trainPredRFC(XTrain, YTrain, XTest, YTest, nEst, mD):
    # Create   
    clf = ensemble.RandomForestClassifier(n_estimators = nEst,
                                          criterion = 'gini',
                                          max_depth = mD,
                                          min_samples_split = 40,
                                          min_samples_leaf = 2,
                                          min_weight_fraction_leaf = 0.0,
                                          max_features = 'auto',
                                          max_leaf_nodes = None,
                                          bootstrap = True,
                                          oob_score = False,
                                          n_jobs = nJobs,
                                          random_state = None,
                                          verbose=1,
                                          warm_start = False,
                                          class_weight = None)
    # Fit                                    
    clf.fit(XTrain, YTrain) 
    # Score   
    yPred, score = scoreMod(clf, XTest, YTest)
    
    return clf, yPred, score



## Importing

rnd=120
random.seed(rnd)

XTrain = pd.read_csv('train.csv')
YTrain = XTrain['TARGET'].values
XTrain = XTrain.drop('TARGET', axis=1)
XTrainIDs  = XTrain['ID']
XTrain = XTrain.drop('ID', axis=1)

XSubmit = pd.read_csv("test.csv")
XSubmitIDs = XSubmit['ID']
XSubmit = XSubmit.drop('ID', axis=1)

## Feature engineering


## Subsetting
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(
     XTrain, YTrain, test_size=0.4, random_state=0)

plt.plot(XTrainIDs)
plt.hist(YTrain)

## Fitting

## Train and test extra trees
#clfExT, yPred, score = trainPredExT(xtrain, ytrain, xtest, ytest, 250, 35)
#clfRFC, yPred, score = trainPredRFC(xtrain, ytrain, xtest, ytest, 250, 50)

nEsts =  range(100,1000,50)
mDs = range(10,100,10)
scoresRFC = np.zeros([len(nEsts),len(mDs)])
scoresExT = np.zeros([len(nEsts),len(mDs)])
times = np.zeros([len(nEsts),len(mDs)])

c=-1
for mD in mDs:
    c += 1
    r =- 1
    for nEst in nEsts:
        r += 1
        print(mD, nEst)
        
        start = time.clock()
        clfRFC, yPred, scoreRFC = trainPredRFC(xtrain, ytrain, xtest, ytest, 
                                               nEst, mD)
        clfExT, yPred, scoreExT = trainPredExT(xtrain, ytrain, xtest, ytest, 
                                               nEst, mD)
        end = time.clock()
        
        scoresRFC[r,c] = scoreRFC
        scoresExT[r,c] = scoreExT
        times[r,c] = end

plt.figure(1)
plt.pcolor(scoresRFC)

plt.figure(4)
plt.pcolor(scoresExT)

## Get highest scores
bestRow = -1
bestCol = -1
scoreMaxExT = np.amax(scoresExT)
scoreMaxRFC = np.amax(scoresRFC)
c = -1
for mD in mDs:
    c += 1
    r = -1
    for nEst in nEsts:
        r +=1
        if scoresExT[r,c] == scoreMaxExT:
            bestRowExT = r
            bestColExT = c
        if scoresRFC[r,c] == scoreMaxRFC:
            bestRowRFC = r
            bestColRFC = c
            
            
bestmDExT = mDs[bestColExT]
bestnEstExT = nEsts[bestRowExT]            
bestmDRFC = mDs[bestColRFC]
bestnEstRFC = nEsts[bestRowRFC]  

print('RFC: mD:', bestmDRFC, ', nEsts:', bestnEstRFC)        
print('ExT: mD:', bestmDExT, ', nEsts:', bestnEstExT)
# RFC: mD: 30, nEsts: 800
# ExT: mD: 70, nEsts: 450


clfRFC, yPred, score = trainPredRFC(xtrain, ytrain, xtest, ytest, 
                                    bestnEstRFC, bestmDRFC)
clfExT, yPred, score = trainPredExT(xtrain, ytrain, xtest, ytest, 
                                    bestnEstExT, bestmDExT)



# yPred = clfExT.predict_proba(XSubmit)

yPredRFC = clfRFC.predict_proba(XSubmit)
yPredExT = clfExT.predict_proba(XSubmit)

plt.figure(5)
plt.hist(yPredRFC[:,1], 150)
plt.figure(5)
plt.hist(yPredExT[:,1], 150)


#submission=pd.read_csv('sample_submission.csv')
#submission.index=submission.ID
#submission.TARGET=yPred[:,1]
#submission.to_csv('pySubmission.csv', index=False)
#submission.PredictedProb.hist(bins=30)


