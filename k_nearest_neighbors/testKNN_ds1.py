import pickle
from sklearn.metrics import accuracy_score

#Import trained model
with open('k_nearest_neighbors_model_ds1.pkl', 'rb') as myFile:
    kNNclassifier = pickle.load(myFile)

#Validation set has class in the last column just like training set 
with open('../DataSet-Release 2/ds1/ds1Test.csv', 'r') as myFile:
    ds1Test = [line.split(',') for line in myFile.read().split('\n')]#[1:] #Remove header, but validation set does not have header
ds1Test.pop()
featuresds1T = [d for d in ds1Test]
featuresds1T = [[int(x) for x in row] for row in featuresds1T] #Convert chars to int

#Test using testing set
validation_predicted = kNNclassifier.predict(featuresds1T)
print (validation_predicted) #Returns ndarray
with open('ds1Test-3.csv', 'w') as myFile:
    for i in range (len(validation_predicted)):
        myFile.write('{},{}\n'.format(i+1, validation_predicted[i]))

