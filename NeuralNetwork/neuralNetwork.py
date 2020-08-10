import math
import numpy as np
import tensorflow as tf

# Exceptions used
class UnknowFunction(Exception) :
    pass

class WrongDimention(Exception) :
    pass


# activation functions
def relu(z) :
    return max(0,z)
    
def sigmoid(z) :
    return 1/(1+math.exp(-z))

def tanh(z) :
    return math.tanh(z)


# utility functions
def convolution2D(data,kernel,stride,padding) :
    kernel = np.flipud(np.fliplr(kernel))
    lenKernel = kernel.shape[0] 
    widKernel = kernel.shape[1] 
    lenData = data.shape[0] 
    widData = data.shape[0]
    lenRes = int(((lenData — lenKernel + 2 * padding) / strides) + 1)
    widRes = int(((widData — widKernel + 2 * padding) / strides) + 1)
    res = np.zeros((lenRes, widRes))
    if padding != 0 :
        dataPadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
        dataPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    else :
        dataPadded = data
    for j in range(data.shape[1]) :
        if j > data.shape[1] — widKernel : 
            break
        if j % strides == 0 :
            for i in range(data.shape[0]) :
                if i > data.shape[0] — lenKernel : 
                    break
                try :
                    if i % strides == 0: 
                        res[i, j] = (kernel * dataPadded[i: i + lenKernel, j: j + widKernel]).sum()
                except :
                    break
    return res

def checkActivation(name) :
    if name == "relu" :
        return lambda z : relu(z)
    elif name == "sigmoid" :
        return lambda z : sigmoid(z)
    elif name == "tanh" :
        return lambda z : tanh(z)
    else
        raise UnknowFunction(name)


# default score
def noScore(restheo, rescomputed) :
    if restheo == rescomputed :
        return 0
    else :
        return 1


# definition of class to create a neural network
class NeuralNetwork :
    """Principal Class to create a neural network"""
    def __init__(self) :
        self.__nbLayer = 0
        self.__layers = []
        self.__resNetWaiting = []

    def add(self,layer) :
        """Add a layer created by user in the NeuralNetwork"""
        self.__layers.append(layer)
        for resNet in self.__resNetWaiting :
            if resNet.insert() :
                self.__layers.append(resNet)
                self.__resNetWaiting.remove(resNet)

    def learn(self,data,result,score = lambda restheo, rescomputed : noScore(restheo, rescomputed)) :
        """compute the learning of the neural network"""
        #count the number of parameter
        nbParameter = 0
        for layer in self.__layers :
            nbParameter = nbParameter + layer.getNbParameter()
        #optimisation
        paramsRes = np.zeros(nbParameter)
        loss = lambda params : self.loss(data,result,score,params)
        #TODO tf.optimisation

    def compute(self,data) :
        """compute the use of the neural network after the learning"""
        for layer in self.__layers :
            data = layer.compute(data)
        return data
    
    def loss(self,data,result,score,params) :
        res = 0
        n = len(label)
        for i in range(n) :
            res = res + score(result[i], self.compute(data[i],params))/n
        return res


# class that define a layer. Each class must have the getNbParameter() and compute(data,param) functions
class DenseLayer :
    """a class representing a dense layer in a neural network"""
    def __init__(self,sizeNeuron,activation) :
        self.__sizeNeuron = sizeNeuron
        self.__activation = np.vectorize(checkActivation(activation))
    
    def getNbParameter(self) :
        return self.__sizeNeuron[0] * self.__sizeNeuron[1]
    
    def compute(self,data,param) :
        x = np.reshape(data,(-1,1))
        if np.size(x,1) != self.__sizeNeuron[1] :
            raise WrongDimention
        neuron = np.reshape(np.array(param[:(self.__sizeNeuron[0]*self.__sizeNeuron[1])]),sizeNeuron)
        linearResult = np.dot(neuron,x)
        return self.__activation(linearResult)


class ResNetLayer :
    def __init__(self, nbContainedLayer) :
        if (not isinstance(nbContainedLayer,int)) or nbContainedLayer < 1 :
            raise WrongNumberOfContainedLayer()
        self.__nbContainedLayer = nbContainedLayer
        self.__x = None
    
    def getNbParameter(self) :
        return 0
    
    def compute(self,data) :
        if self.__x == None :
            self.__x = data
            return data
        else :
            return data + self.__x

    def insert(self) :
        self.__nbContainedLayer = self.__nbContainedLayer -1
        if self.__nbContainedLayer == 0 :
            return True
        else :
            return False


class ConvLayer :
    """a class representig a convutionnal layer in a neural network"""
    def __init__(self,kernelSize,stride=1,padding=0,activation) :
        kernelSizeList = list(kernelSize)
        for i in range(len(kernelSizeList),4)
            kernelSizeList.append(1)
        self.__kernelSize = tuple(kernelSizeList)
        self.__stride = stride
        self.__padding = padding
        self.activation = np.vectorize(checkActivation(activation))
        self.__nbParameter = 1
        for dimension in kernelSize :
            self.__nbParameter = self.nbParameter * dimension
        self.__sizeInput = None
        self.__kernel = None
    
    def getNbParameter(self) :
        return self.__nbParameter

    def compute(self,data,param) :
        if sizeInput != None :
            if data.shape[0] != self.__sizeInput[0] or data.shape[1] != self.__sizeInput[1] or data.shape[2] != self.__sizeInput[2] :
                raise WrongDimention
        else :
            if data.shape[0] < self.__kernelSize[0] or data.shape[1] < self.__kernelSize[1] or data.shape[2] != self.__kernelSize[2] :
                raise WrongDimention
        if self.__kernel = None :
            kernel = np.zeros(self.__kernelSize)
            index = 0
            for i in range(self.__kernelSize[0]) :
                for j in range(self.__kernelSize[1]) :
                    for k in range(self__kernelSize[2]) :
                        for l in range(self.__kernelSize[3]) :
                            kernel[i,j,k,l] = param[index]
                            index = index + 1
        else :
            kernel = self.__kernel
        linearRes = np.zeros(((data.shape[0] + 2*self.__padding - kernel.shape[0])/self.__stride + 1,(data.shape[1] + 2*self.__padding - kernel.shape[1])/self.__stride + 1,kernel.shape[3]))
        for k in range(kernel.shape[3]) :
            for c in range(kernel.shape[2]) :
                linearRes[:,:,k] = linearRes[:,:,k] + convolution2D(data[:,:,c],kernel[:,:,c,k],self.__stride,self.__padding)
        return activation(linearRes)
    
    def setSizeInput(self,sizeInput) :
        """fonction used to verify the input at the enter of the layer"""
        if sizeInput[0] < self.__kernelSize[0] or sizeInput[1] < self.__kernelSize[1] or sizeInput[2] != self.__kernelSize[2] :
            raise WrongDimention
        self.__sizeInput = sizeInput

    def setKernel(self,kernel) :
        """fonction to set the kernel"""
        self.__kernel = kernel
        self.__nbParameter = 0


class MaxPooling :
    """layer that perform a MaxPooling"""
    def __init__(self,size) :
        self.__size = size

    def getNbParameter(self) :
        return 0
    
    def compute(self,data,param) :
        #TODO
        pass


class RecurentNetwork :
    #TODO
    def __init__(self) :
        pass

    def getNbParameter(self) :
        pass

    def compute(self) :
        pass
