import numpy as np
import tensorflow as tf
from sys import path
path.append("..")
import NeuralNetwork

tf.compat.v1.enable_eager_execution()

## Convolution test
A = np.array(
    [[ 1, 1, 0, 1]
    ,[ 0, 0, 0, 1]
    ,[ 1, 1, 1, 0]
    ,[ 1, 0, 0, 1]])
B = np.array(
    [[-1,-2,-1]
    ,[ 0, 0, 0]
    ,[ 1, 2, 1]])
C = np.array(
    [[ 3, 1, 3, 5, 3, 3]
    ,[ 2, 2, 8, 8, 3, 9]
    ,[ 3, 4, 7, 7, 2, 7]
    ,[ 5, 3, 6, 8, 4, 7]
    ,[ 3, 8, 8, 5, 7, 4]
    ,[ 7, 9, 6, 4, 6, 9]])

D = NeuralNetwork.convolution2D(tf.constant(A,dtype="double"),tf.constant(B,dtype="double"))
test = ((D.numpy()) == np.array(
    [[ 1, 1]
    ,[ 1, 0]]))
assert((test.all()))

D = NeuralNetwork.convolution2D(tf.constant(A,dtype="double"),tf.constant(B,dtype="double"),1,1)
test = ((D.numpy()) == np.array(
    [[ 0, 0, 1, 2]
    ,[ 0, 1, 1,-1]
    ,[ 2, 1, 0, 0]
    ,[-3,-4,-3,-1]]))
assert(test.all())

D = NeuralNetwork.convolution2D(tf.constant(C,dtype="double"),tf.constant(B,dtype="double"),2,0)
test = ((D.numpy()) == np.array(
    [[10, 7]
    ,[ 9, 2]]))
assert(test.all())


## DenseLayer test
try :
    layer = NeuralNetwork.DenseLayer((2,3),"relu")
    assert(False)
except (NeuralNetwork.UnusableType) :
    layer = NeuralNetwork.DenseLayer(3,"relu")

data = np.array([ 2,-3,-4,-5, 6, 7, 8,-9])
param = np.array([4,5,6,7,8,9,10,11,-12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

layer.setInputSize(len(data))

assert(layer.getNbParameter() == len(param))

test = (layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() == np.array([26,0,58]))
assert(test.all())


## ConvolutionLayer test
layer = NeuralNetwork.ConvLayer((1,2,3,3),"relu")

data = np.array([
    [[1, 1, 0, 1]
    ,[0, 0, 0, 1]
    ,[1, 1, 1, 0]
    ,[1, 0, 0, 1]]
    ,
    [[1, 0, 1, 1]
    ,[2, 3, 1, 2]
    ,[1, 1, 1, 1]
    ,[0, 2, 0, 0]]])
    
param = np.array([-1,-2,-1,0,0,0,1,2,1,-1,-2,-1,0,0,0,1,2,1])

layer.setInputSize(data.shape)

assert(layer.getNbParameter() == len(param))

test = (layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() == np.array([
    [[3.,2.]
    ,[0.,0.]]]))
assert(test.all())
assert(test.shape == layer.getOutputSize())

layer.setKernel(np.reshape(param,(2,3,3)))

assert(layer.getNbParameter() == 0)

test = (layer.compute(tf.constant(data,dtype="double"),[]).numpy() == np.array(
    [[[3,2]
    ,[0,0]]]))
assert(test.all())
assert(test.shape == layer.getOutputSize())


## MaxPooling test
E = np.array([
    [[ 12, 20, 30,  0]
    ,[  8, 12,  2,  0]
    ,[ 34, 70, 37,  4]
    ,[112,100, 25, 12]]])
pooling = NeuralNetwork.MaxPooling((2,2))
pooling.setInputSize(E.shape)

pooling.setStride(2)
D = pooling.compute(tf.constant(E,dtype="double"),[]).numpy()
test = (D == np.array([
    [[ 20, 30]
    ,[112, 37]]]))
assert(test.all())
assert(test.shape == layer.getOutputSize())

pooling.setStride(1)
D = pooling.compute(tf.constant(E,dtype="double"),[]).numpy()
test = (D == np.array(
    [[[ 20, 30, 30]
    ,[ 70, 70, 37]
    ,[112,100, 37]]]))
assert(test.all())
assert(test.shape == pooling.getOutputSize())


## AveragePooling test
pooling = NeuralNetwork.AveragePooling((2,2))
pooling.setInputSize(E.shape)

pooling.setStride(2)
D = pooling.compute(tf.constant(E,dtype="double"),[]).numpy()
test = (D == np.array([
    [[   13,   8]
    ,[   79,19.5]]]))
assert(test.all())
assert(test.shape == pooling.getOutputSize())


## test SimpleRNN
layer = NeuralNetwork.SimpleRNN(3)

data = np.array(
    [[1,1]
    ,[1,0]])

layer.setInputSize(data.shape)

param = np.array([1,2, 1,1,1,2, 0,-1,1,0, 0,1,1,2])

assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1]
    ,[1, 1]
    ,[1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())

layer.setOutputSize((2,3))

param = np.array([1,2, 1,1,1,2, 0,-1,1,0, 0,1,1,2,2,0])
assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1, 1]
    ,[1, 1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())


## test LSTM
layer = NeuralNetwork.LSTM(3)

data = np.array(
    [[1,1]
    ,[1,0]])

layer.setInputSize(data.shape)

param = np.array([1,2, 1,1,1,2, 0,-1,1,0, 0,1,1,2, 1,0,1,1, -1,-1,0,-1, 1,1,0,1, 1,-1,0,1, 0,1,0,0, 0,1,0,2])
assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1]
    ,[1, 1]
    ,[1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())

layer.setOutputSize((2,3))

param = np.array([1,2, 1,1,1,2, 0,-1,1,0, 0,1,1,2, 1,0,1,1, -1,-1,0,-1, 1,1,0,1, 1,-1,0,1, 0,1,0,0, 0,1,0,2,1,0])
assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1, 1]
    ,[1, 1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())


## test GRU
layer = NeuralNetwork.GRU(3)

data = np.array(
    [[1,1]
    ,[1,0]])

layer.setInputSize(data.shape)

param = np.array([1,2, 1,1,1,2, 1,0,1,1, -1,-1,0,-1, 1,1,0,1, 1,-1,0,1, 0,1,0,0, 0,1,0,2])
assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1]
    ,[1, 1]
    ,[1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())

layer.setOutputSize((2,3))

param = np.array([1,2, 1,1,1,2, 1,0,1,1, -1,-1,0,-1, 1,1,0,1, 1,-1,0,1, 0,1,0,0, 0,1,0,2,1,-1])
assert(layer.getNbParameter() == len(param))

# I work with float, so I use inequality
test = ((layer.compute(tf.constant(data,dtype="double"),tf.constant(param,dtype="double")).numpy() - np.array(
    [[1, 1, 1]
    ,[1, 1, 1]]) < 0.00000001))
assert(test.all())
assert(test.shape == layer.getOutputSize())
