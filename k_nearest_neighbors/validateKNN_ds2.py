import pickle
from sklearn.metrics import accuracy_score

#Import trained model
with open('k_nearest_neighbors_model_ds2.pkl', 'rb') as myFile:
    kNNclassifier = pickle.load(myFile)

#Validation set has class in the last column just like training set 
with open('../DataSet-Release 1/ds2/ds2Val.csv', 'r') as myFile:
    ds2Val = [line.split(',') for line in myFile.read().split('\n')]#[1:] #Remove header, but validation set does not have header
ds2Val.pop()
featuresds2V = [d[:-1] for d in ds2Val]
featuresds2V = [[int(x) for x in row] for row in featuresds2V] #Convert chars to int
labelsds2V = [d[-1] for d in ds2Val]
labelsds2V = [int(x) for x in labelsds2V] #Convert chars to int

#Test using validation set
validation_predicted = kNNclassifier.predict(featuresds2V)
with open('ds2Val-3.csv', 'w') as myFile:
    for i in range (len(validation_predicted)):
        myFile.write('{},{}\n'.format(i+1, validation_predicted[i]))

#proper way to calculate accuracy
accuracy = accuracy_score(labelsds2V, validation_predicted)
print (accuracy)
