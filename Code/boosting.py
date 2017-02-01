__author__ ="vivek, sasi, himanshu, aravind"
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def loadData():
	subsetDate = pd.read_csv("train_date.csv", chunksize=100000,index_col=0, dtype=np.float32)
	subsetNum = pd.read_csv("train_numeric.csv",chunksize=100000, usecols=list(range(969)),index_col=0, dtype=np.float32)
	lst=[]
	for i,j in zip(subsetDate,subsetNum):
		lst.append(pd.concat([i,j]).sample(frac=0.05))
		featValues=pd.concat(lst)
	del lst
	labels = pd.read_csv("train_numeric.csv", dtype=np.float32, usecols=[0,969],index_col=0)
	labels=labels.loc[featValues.index]
	labels=labels.values.ravel()
	featValues = featValues.values
	#print featValues
	return featValues,labels


def classify(sigFeat):
	dateFeat = 1156
	sigcolDate = sigFeat[sigFeat<dateFeat]+1
	colsDate = np.concatenate([[0],sigcolDate])
	sigcolNum = sigFeat[sigFeat>=dateFeat]+1-dateFeat
	colsNum = np.concatenate([[0],sigcolNum])
	#print colsNum,colsDate
	date = pd.read_csv("train_date.csv",index_col=0,dtype=np.float32,usecols=colsDate)
	num = pd.read_csv("train_date.csv",index_col=0,dtype=np.float32,usecols=colsNum)
	del sigcolDate,sigcolNum
	updatedFeat = np.concatenate([date.values,num.values],axis=1)
	del date,num
	updatedLabels = pd.read_csv("train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969])
	updatedLabels=updatedLabels.values.ravel()
	print "Labels Updated"
	updatedCl = XGBClassifier(max_depth=5, base_score=0.005,objective='binary:logistic',colsample_bytree=0.5)
	crossValid = StratifiedKFold(updatedLabels, n_folds=3)
	predictLabels = np.ones(updatedLabels.shape[0])
	for (train, test) in crossValid:
		predictLabels[test] = updatedCl.fit(updatedFeat[train], updatedLabels[train]).predict_proba(updatedFeat[test])[:,1]
		print "ROC:", roc_auc_score(updatedLabels[test], predictLabels[test])
	try:
		np.savetxt('actual.txt',updatedLabels[test],newline="\n")
		np.savetxt('predict.txt',predictLabels[test], newline="\n")
	except:
		print " "
	uDate = pd.read_csv("test_date.csv", index_col=0, dtype=np.float32,usecols=colsDate)
	uNum = pd.read_csv("test_numeric.csv", index_col=0, dtype=np.float32,usecols=colsNum)
	uFeat = np.concatenate([uDate.values,uNum.values], axis=1)
	del colsDate, colsNum, uDate, uNum
	uPr = (updatedCl.predict_proba(uFeat)[:,1] > 0.19).astype(np.int8)
	print "predicted labels: ", uPr
	try:
		result = pd.read_csv('submission.csv',index_col=0)
		result["Response"] = uPr
	except:
		try:
			np.savetxt("submission.csv", np.dstack((np.arange(1, uPr.size+1),np.around(uPr)))[0],"%d,%d",header="Id,Response")
		except:
			print " "

def sigFeatures(featValues,labels):
	sigCl = XGBClassifier(base_score=0.005,max_depth=3)
	sigCl.fit(featValues, labels)
	sigFeat = np.where(sigCl.feature_importances_>0.0047)[0]
	#print(sigFeat)
	return sigFeat


if __name__ == '__main__':
	loadResult=loadData()
	print "features: ", loadResult[0]
	print "labels: ", loadResult[1]
	print "#############"
	sigFeat=sigFeatures(loadResult[0],loadResult[1])
	print "important feature indexes: ", sigFeat
	print "########"
	predict=classify(sigFeat)
