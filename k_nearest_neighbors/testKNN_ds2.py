import pickle
from sklearn.metrics import accuracy_score

#Import trained model
with open('k_nearest_neighbors_model_ds2.pkl', 'rb') as myFile:
    kNNclassifier = pickle.load(myFile)

#Validation set has class in the last column just like training set 
with open('../DataSet-Release 2/ds2/ds2Test.csv', 'r') as myFile:
    ds2Test = [line.split(',') for line in myFile.read().split('\n')]#[1:] #Remove header, but validation set does not have header
ds2Test.pop()
featuresds2T = [d for d in ds2Test]
featuresds2T = [[int(x) for x in row] for row in featuresds2T] #Convert chars to int

#Test using validation set
validation_predicted = kNNclassifier.predict(featuresds2T)
with open('ds2Test-3.csv', 'w') as myFile:
    for i in range (len(validation_predicted)):
        myFile.write('{},{}\n'.format(i+1, validation_predicted[i]))

