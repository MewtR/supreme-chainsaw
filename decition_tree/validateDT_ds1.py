import pickle
from sklearn.metrics import accuracy_score

#Import trained model
with open('decision_tree_model_ds1.pkl', 'rb') as myFile:
    dTclassifier = pickle.load(myFile)

#Validation set has class in the last column just like training set 
with open('../DataSet-Release 1/ds1/ds1Val.csv', 'r') as myFile:
    ds1Val = [line.split(',') for line in myFile.read().split('\n')]#[1:] #Remove header, but validation set does not have header
ds1Val.pop()
featuresds1V = [d[:-1] for d in ds1Val]
featuresds1V = [[int(x) for x in row] for row in featuresds1V] #Convert chars to int
labelsds1V = [d[-1] for d in ds1Val]
labelsds1V = [int(x) for x in labelsds1V] #Convert chars to int

#Test using validation set
validation_predicted = dTclassifier.predict(featuresds1V)
print (validation_predicted) #Returns ndarray
#print (validation_predicted.__class__.__name__)
print (labelsds1V) # Is a list
#print (labelsds1V.__class__.__name__)
with open('ds1Val-dt.csv', 'w') as myFile:
    for i in range (len(validation_predicted)):
        myFile.write('{},{}\n'.format(i+1, validation_predicted[i]))

#proper way to calculate accuracy
accuracy = accuracy_score(labelsds1V, validation_predicted)
print (accuracy)
