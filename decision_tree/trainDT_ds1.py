import pickle
from sklearn import tree

with open('../DataSet-Release 1/ds1/ds1Train.csv', 'r') as myFile:
    ds1Train = [line.split(',') for line in myFile.read().split('\n')]#[1:]#Remove header, but training set does not have header

ds1Train.pop()
#print (ds1Train[0])
featuresds1T = [d[:-1] for d in ds1Train]
featuresds1T = [[int(x) for x in row] for row in featuresds1T] #Convert chars to int
labelsds1T = [d[-1] for d in ds1Train]
labelsds1T = [int(x) for x in labelsds1T] #Convert chars to int
#print (featuresds1T) #List: everything but last column for every row
#print (labelsds1T)

#Train using training set
#dTclassifier = tree.DecisionTreeClassifier()
dTclassifier = tree.DecisionTreeClassifier(criterion='entropy')
dTclassifier.fit(featuresds1T, labelsds1T)

#Export trained model
with open('decision_tree_model_ds1.pkl', 'wb') as myFile:
    pickle.dump(dTclassifier, myFile)

