import numpy
import pandas
import seaborn
from matplotlib import pyplot

def featureDependency(features):
    lFeatures = {}
    sFeatures = {}
    tmpLine=[]
    tmpStation=[]
    for f in features:
        tmpLine.append(f.split('_')[0])
        tmpStation.append(f.split('_')[1])   
    lines = set(tmpLine)
    stations = set(tmpStation)
    for l in lines:
        tmp=[]
        for f in features:
            if l+'_' in f:
                tmp.append(f)
        lFeatures[l] = tmp
    for s in stations:
        tmp=[]
        for f in features:
            if s+'_' in f:
                tmp.append(f)
        sFeatures[s] = tmp        
    return lFeatures, sFeatures

def visual(sData):
    pyplot.figure(figsize=(20,8))
    pyplot.title('Error Rate between Production Stations')
    pyplot.xlabel('Station Error Rate')
    seaborn.barplot(x=sData.index.values,y='Error_Rate',  data=sData, color="blue")
    pyplot.show()
	
def stationData(sFeatures):
    sError = []
    for s in sFeatures:
        cols = ['Id', 'Response']
        cols.extend(sFeatures[s])
        dataFrame = pandas.io.parsers.read_csv("train_numeric.csv", usecols=cols,nrows=10000).dropna(subset=sFeatures[s], how='all')
        responseList = [dataFrame[dataFrame.Response == 1].size,float(dataFrame[dataFrame.Response == 0].size)]
        errorRate = responseList[0]/(responseList[1])
        sError.append([dataFrame.shape[0],dataFrame.shape[1]-2, errorRate])
    sData = pandas.DataFrame(sError, index=sFeatures, columns=['Samples', 'Features', 'Error_Rate']).sort_index()
    return sData

trainNumericDataFeatures = pandas.io.parsers.read_csv("train_numeric.csv", nrows=1).drop(['Id', 'Response'], axis=1).columns.values
lFeatures, sFeatures = featureDependency(trainNumericDataFeatures)
sData=stationData(sFeatures)
visual(sData)
