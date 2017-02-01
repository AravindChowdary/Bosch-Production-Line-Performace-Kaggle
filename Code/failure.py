import numpy as np
import pandas as pd

y = pd.read_csv("submission.csv", index_col=0)
response = y['Response']
count_0 =0
count_1 =0
for i in response:
	if i ==0:
		count_0+=1
	else:
		count_1+=1
		
percentFailure = float(count_1)/(float(count_1)+float(count_0))
#print (float(count_1)+float(count_0))
print "Percentage Failure:", percentFailure*100