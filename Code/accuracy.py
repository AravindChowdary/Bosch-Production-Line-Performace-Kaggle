import numpy
predvalue=numpy.loadtxt("predict.txt")
intPred=[0]*len(predvalue)
for i in range(len(predvalue)):
    if predvalue[i] < 0.0038:
        intPred[i] = 1

intAct=numpy.loadtxt("actual.txt")
#print intAct

count=0
for i,j in zip(intPred,intAct):
    if i==j:
        count+=1
accuracy = float(count)/float(len(intPred))
print "accuracy: ", accuracy*100
