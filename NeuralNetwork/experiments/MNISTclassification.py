import numpy as np
from sys import path
path.append("..")
import NeuralNetwork

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
data_train = np.zeros((X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
label_train = np.zeros((Y_train.shape[0],10))
for i in range(X_train.shape[0]) :
    data_train[i,0,:,:] = X_train[i,:,:]
    label_train[i,Y_train[i]] = 1

data_test = np.zeros((2,1,X_test.shape[1],X_test.shape[2]))
data_test[0,0,:,:] = X_test[0,:,:]
data_test[1,0,:,:] = X_test[1,:,:]

classifier = NeuralNetwork.Network((1,28,28))

layer1 = NeuralNetwork.ConvLayer((30,1,5,5),"relu")
layer2 = NeuralNetwork.MaxPooling((2,2))
layer3 = NeuralNetwork.ConvLayer((15,30,3,3),"relu")
layer4 = NeuralNetwork.MaxPooling((2,2))
layer5 = NeuralNetwork.FlattenLayer()
layer6 = NeuralNetwork.DenseLayer(128,"relu")
layer7 = NeuralNetwork.DenseLayer(50,"relu")
layer8 = NeuralNetwork.DenseLayer(10,"tanh")

classifier.add(layer1)
classifier.add(layer2)
classifier.add(layer3)
classifier.add(layer4)
classifier.add(layer5)
classifier.add(layer6)
classifier.add(layer7)
classifier.add(layer8)

classifier.learn(data_train[:10,:,:,:],label_train[:10,:])

for i in range(data_test.shape[0]) :
    res = classifier.compute(data_test[i,:,:,:])
    print("le nombre " + str(Y_test[0]) + " a pour resultat " + str(res))