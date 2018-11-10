import pickle
from sklearn import naive_bayes

with open('../DataSet-Release 1/ds2/ds2Train.csv', 'r') as myFile:
    ds2Train = [line.split(',') for line in myFile.read().split('\n')]#[1:]#Remove header, but training set does not have header

ds2Train.pop()
#print (ds2Train[0])
featuresds2T = [d[:-1] for d in ds2Train]
featuresds2T = [[int(x) for x in row] for row in featuresds2T] #Convert chars to int
labelsds2T = [d[-1] for d in ds2Train]
labelsds2T = [int(x) for x in labelsds2T] #Convert chars to int

#Train using training set
nBclassifier = naive_bayes.BernoulliNB()
nBclassifier.fit(featuresds2T, labelsds2T)

#Export trained model
with open('naive_bayes_model_ds2.pkl', 'wb') as myFile:
    pickle.dump(nBclassifier, myFile)

