# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:00:47 2016

@author: Gareth
"""
#%reset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from sklearn import ensemble
from sklearn.metrics import auc, roc_curve
from math import log

## Set seed
rnd = 1234567

## Import and drop ID/targets
XTrain = pd.read_csv('train.csv')
YTrain = XTrain['TARGET'].values
XTrain = XTrain.drop('TARGET', axis=1)
XTrainIDs  = XTrain['ID']
XTrain = XTrain.drop('ID', axis=1)

XSubmit = pd.read_csv("test.csv")
XSubmitIDs = XSubmit['ID']
XSubmit = XSubmit.drop('ID', axis=1)


## Fix var3 and 9999999999 error codes
# var3
idxTrain = XTrain['var3'] == -999999
idxSubmit = XSubmit['var3'] == -999999
print('Train var3:', sum(idxTrain), '-999999s')
print('Submit var3:', sum(idxSubmit), '-999999s')
XTrain.loc[idxTrain, 'var3'] = 0
XSubmit.loc[idxSubmit, 'var3'] = 0

# 9999999999 code in other vars
for c in XTrain.columns:
    XTrain.loc[XTrain[c]==9999999999, c] = 0
    XSubmit.loc[XSubmit[c]==9999999999, c] = 0


## Add zero count
XTrain.insert(1, 'nZeros', (XTrain == 0).astype(int).sum(axis=1))
XSubmit.insert(1, 'nZeros', (XSubmit == 0).astype(int).sum(axis=1))


## Get Age column (var15) and other "happy columns?"
XTrainAge = XTrain['var15']
XSubmitAge = XSubmit['var15']
SMV5H2 = XSubmit['saldo_medio_var5_hace2']
SV33 = XSubmit['saldo_var33']
var38 = XSubmit['var38']
V21 = XSubmit['var21']

## Log var38
XTrain['var38'] = log(XTrain['var38'])
XSubmit['var38'] = log(XSubmit['var38'])

## Remove duplicate features
remove = [] # Prepare list of features to drop
c = XTrain.columns
# For all columns (except last)
for ci in range(0, len(c)-1):
    # Get column
    c1 = XTrain[c[ci]].values
    # For remaining columns (including last)
    for cj in range(ci+1, len(c)):
        # Get column
        c2 = XTrain[c[cj]].values
        
        # Compare columns using np.array_equal
        if np.array_equal(c1, c2):
            print(c[ci], c[cj], "True")
            # Append second column to list to remove
            remove.append(c[cj])
        else:
            print(c[ci], c[cj], "False")
            
# remove is list containing column names. 
# May contain duplicates, but doesn't matter.            
XTrain.drop(remove, axis=1, inplace=True)
XSubmit.drop(remove, axis=1, inplace=True)

## Remove empty predictors (based on std)
remove = []
c = XTrain.columns
# For all columns
for ci in range(len(c)-1):
    # print(XTrain[c[ci]].std())
    # Is std 0?    
    if XTrain[c[ci]].std() == 0:
        # Append to list to remove
        remove.append(c[ci])

XTrain.drop(remove, axis=1, inplace=True)
XSubmit.drop(remove, axis=1, inplace=True)

## Limit var scale
for c in XTrain.columns:
    # Min
    lim = XTrain[c].min()
    # "Chained indexing" - bad
    # XSubmit[c][XSubmit[c]<lim] = lim
    # Use df.loc instead
    XSubmit.loc[XSubmit[c]<lim,c] = lim
    
    # Max
    lim = XTrain[c].max()
    XSubmit.loc[XSubmit[c]>lim,c] = lim
    

## Subset
# Train on full training set here
    
## Prepare models to train - just add previously found parameters here for now
nJobs = 6;
def trainPredExT(XTrain, YTrain, XTest, YTest, nEst, mD):
    # Create    
    clf = ensemble.ExtraTreesClassifier(n_estimators = nEst,
                                        max_features = 'auto',
                                        criterion= 'entropy',
                                        min_samples_split= 40,
                                        max_depth = mD,
                                        min_samples_leaf= 2, 
                                        n_jobs = nJobs, 
                                        random_state = rnd,
                                        verbose = 1)
    # Fit                                    
    clf.fit(XTrain, YTrain) 
    # Score   
    yPred, score = scoreMod(clf, XTest, YTest)
    return clf, yPred, score
    
def trainPredRFC(XTrain, YTrain, XTest, YTest, nEst, mD):
    # Create   
    clf = ensemble.RandomForestClassifier(n_estimators=nEst,
                                          criterion = 'gini',
                                          max_depth = mD,
                                          min_samples_split = 2,
                                          min_samples_leaf = 2,
                                          min_weight_fraction_leaf = 0.0,
                                          max_features = 'auto',
                                          max_leaf_nodes = None,
                                          bootstrap = True,
                                          oob_score = False,
                                          n_jobs = nJobs,
                                          random_state = None,
                                          verbose = 1,
                                          warm_start = False,
                                          class_weight = None)
    # Fit                                    
    clf.fit(XTrain, YTrain) 
    # Score   
    yPred, score = scoreMod(clf, XTest, YTest)
    return clf, yPred, score
    
def scoreMod(clf, XTest, YTest):
  
    yPred = clf.predict_proba(XTest) 
    
    if YTest == []:
        # No Y labels available for test, skip
        score = []
    else:
        # Calculate fpr, tpr, and AUC         
        fpr, tpr, _ = roc_curve(YTest, yPred[:, 1])
        plt.plot(fpr,tpr)
        score = auc(fpr, tpr)
        print(score)                          
    return yPred, score    

                        
## Train and RFC and test extra trees
clfExT, yPredExT, score = trainPredExT(XTrain, YTrain, XSubmit, 
                                           [], 450, 70)
clfRFC, yPredRFC, score = trainPredRFC(XTrain, YTrain, XSubmit, 
                                       [], 800, 30)

# Discard second column
yPredExT = yPredExT[:,1]
yPredRFC = yPredRFC[:,1]

plt.hist(yPredExT, 150)
plt.hist(yPredRFC, 150)

## Apply "happy columns"
# Age
yPredExT[XSubmitAge<23] = 0
yPredRFC[XSubmitAge<23] = 0
yPredExT[SMV5H2>160000] = 0
yPredRFC[SMV5H2>160000] = 0
yPredExT[SV33>0] = 0
yPredRFC[SV33>0] = 0
yPredExT[var38 > 3988596] = 0
yPredRFC[var38 > 3988596] = 0
yPredExT[V21>7500]= 0
yPredRFC[V21>7500]= 0

## Write submissions
submission=pd.read_csv('sample_submission.csv')
submission.index = submission.ID
# ExT
submission.TARGET = yPredExT
submission.to_csv('pySubmissionExT.csv', index=False)
# RFC
submission.TARGET = yPredRFC
submission.to_csv('pySubmissionRFC.csv', index=False)
