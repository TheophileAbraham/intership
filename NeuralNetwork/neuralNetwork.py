import math
import numpy as np
from scipy.optimize import fmin

# Exceptions used
class UnknowFunction(Exception) :
    pass

class WrongDimention(Exception) :
    pass

class UnusableType(Exception) :
    pass


# activation functions
def relu(z) :
    return max(0,z)
    
def sigmoid(z) :
    return 1/(1+math.exp(-z))

def tanh(z) :
    return math.tanh(z)


# utility functions
def convolution2D(data,kernel,stride=1,padding=0) :
    """perform a 2D convolution between a matrix and a kernel"""
    lenKernel = kernel.shape[0] 
    widKernel = kernel.shape[1] 
    lenData = data.shape[0] 
    widData = data.shape[1]
    lenRes = int(((lenData - lenKernel + 2 * padding) / stride) + 1)
    widRes = int(((widData - widKernel + 2 * padding) / stride) + 1)
    res = np.zeros((lenRes, widRes))
    if padding != 0 :
        dataPadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
        dataPadded[padding:(-1 * padding), padding:(-1 * padding)] = data
    else :
        dataPadded = data
    for j in range(0, dataPadded.shape[1] - widKernel + 1, stride) :
        for i in range(0, dataPadded.shape[0] - lenKernel + 1, stride) :
            res[i//stride, j//stride] = (kernel * dataPadded[i: i + lenKernel, j: j + widKernel]).sum()
    return res

def softmax(z) :
    expz = np.exp(z)
    return expz/(expz.sum())

def checkActivation(name) :
    """check if the name corespond to an existing activation function and return the right function"""
    if name == "relu" :
        return lambda z : relu(z)
    elif name == "sigmoid" :
        return lambda z : sigmoid(z)
    elif name == "tanh" :
        return lambda z : tanh(z)
    else :
        raise UnknowFunction(name)


# default score
def noScore(restheo, rescomputed) :
    """the score used if no custom score has been inputed by the user"""
    if restheo == rescomputed :
        return 0
    else :
        return 1


# definition of class to create a neural network
class NeuralNetwork :
    """Principal Class to create a neural network"""
    def __init__(self,inputSize) :
        self.__inputSize = inputSize
        self.__nbLayer = 0
        self.__layers = []
        self.__resNetWaiting = []

    def add(self,layer) :
        """Add a layer created by user in the NeuralNetwork"""
        # set the size of the input from the last layer
        if (len(self.__layers) == 0) :
            layer.setInputSize(self.__inputSize)
        else :
            layer.setInputSize(self.__layers[-1].getOutputSize())
        # add the layer to the list of layers that compose the neural network
        self.__layers.append(layer)
        # if the layer is a resNet layer, we store it to add it when it is used the second time
        if isinstance(layer, ResNetLayer) :
            self.__resNetWaiting.append(layer)
        # add the second part of each resNet layer if needed
        for resNet in self.__resNetWaiting :
            if resNet.insert() :
                self.__layers.append(resNet)
                self.__resNetWaiting.remove(resNet)

    def learn(self,data,result,score = lambda restheo, rescomputed : noScore(restheo, rescomputed)) :
        """compute the learning of the neural network"""
        # count the number of parameter
        nbParameter = 0
        for layer in self.__layers :
            nbParameter = nbParameter + layer.getNbParameter()
        # optimisation of the parameters
        paramsRes = np.zeros(nbParameter)
        loss = lambda params : self.loss(data,result,score,params)
        self.__param = fmin(loss,np.zeros(nbParameter))

    def compute(self,data) :
        """compute the use of the neural network after the learning"""
        # for each layer, we give the inner data between the previous layer and the one we work on it
        # and the list of parameters of the neural network, amputed from their first parameters, in a
        # way that the first parameters of the given list is the parameters of the layer
        index = 0
        for layer in self.__layers :
            data = layer.compute(data,param[index:])
            index = index + layer.getNbParameter()
        return data
    
    def loss(self,data,result,score,params) :
        """describe the loss needed for the optimisation"""
        # calcul the mean of the score of each computation of one data and the score needed
        res = 0
        n = len(label)
        for i in range(n) :
            res = res + score(result[i], self.compute(data[i,:,:],params))/n
        return res


# class that define a layer. Each class must have the getNbParameter() and compute(data,param) functions
class DenseLayer :
    """a class representing a dense layer in a neural network"""
    def __init__(self,lengthOutput,activation) :
        # size of the output of the layer, given by the user
        if not(isinstance(lengthOutput,int)) :
            raise UnusableType
        self.__sizeOutput = lengthOutput
        # chose what activation function the user want to use
        self.__activation = np.vectorize(checkActivation(activation))
    
    def getNbParameter(self) :
        return self.__sizeNeuron[0] * self.__sizeNeuron[1]
    
    def setInputSize(self,inputSize) :
        # compute the number of input
        inputLength = 1
        for i in inputSize :
            inputLength = inputLength * i
        # the size of the matrix representing the layer
        self.__sizeNeuron = (self.__sizeOutput,inputLength)
    
    def getOutputSize(self) :
        return self.__sizeOutput
    
    def compute(self,data,param) :
        # vectorize the input
        x = np.reshape(data,(-1,1))
        # construct the matrix representing the layer from the list of parameters
        neuron = np.reshape(np.array(param[:(self.__sizeNeuron[0]*self.__sizeNeuron[1])]),self.__sizeNeuron)
        # compute the result before the activation
        linearResult = np.dot(neuron,x)
        # compute the result and return it after the activation
        return np.reshape(self.__activation(linearResult),(self.__sizeOutput))


class ResNetLayer :
    def __init__(self, nbContainedLayer) :
        # check if nbContainedLayer is usable by the algorithm
        if (not isinstance(nbContainedLayer,int)) or nbContainedLayer < 1 :
            raise WrongNumberOfContainedLayer()
        # it is an integer representing the number of layer between when the layer pick up the
        # inner data and the time it add it to the new inner data
        self.__nbContainedLayer = nbContainedLayer
        # the memory of the data micked up
        self.__x = None

    
    def getNbParameter(self) :
        # this layer didn't take any parameter
        return 0
    
    def setInputSize(self, inputSize) :
        self.__sizeOutput = inputSize
    
    def getOutputSize(self) :
        return self.__sizeOutput
    
    def compute(self,data) :
        # chose if it is the first time this layer is used, or the second
        if self.__x == None :
            # if it is the firt time the layer is called, it have to pick up the inner data, and
            # give it to the following layers
            self.__x = data
            return data
        else :
            # if it is the second time the layer is called, it have to add the inner data picked
            # up when it was first called and add it to the current inner data, and give the result
            # to the following layers
            return data + self.__x

    def insert(self) :
        # check if there is the right number of layers between the moment the layer pick up the inner
        # data and the moment it add it with the current inner data. At each call of this function, the
        # count is decrementing, and if the count is going to zero, the layer is readded in the list of
        # layer of the neural network
        self.__nbContainedLayer = self.__nbContainedLayer -1
        if self.__nbContainedLayer == 0 :
            return True
        else :
            return False


class ConvLayer :
    """a class representig a convutionnal layer in a neural network"""
    def __init__(self,kernelSize,activation) :
        # transform the tuple kernelSize on a tuple of length 4
        kernelSizeList = list(kernelSize)
        for i in range(len(kernelSizeList),4) :
            kernelSizeList.append(1)
        self.__kernelSize = tuple(kernelSizeList)
        # chose what activation function the user want to use
        self.__activation = np.vectorize(checkActivation(activation))
        # calculate the number of parameters needed
        self.__nbParameter = 1
        for dimension in kernelSize :
            self.__nbParameter = self.__nbParameter * dimension
        # the kernel used in the convolution
        self.__kernel = np.array([None])
        # the parameter of the convolution
        self.__stride = 1
        self.__padding = 0
    
    def getNbParameter(self) :
        return self.__nbParameter
    
    def setInputSize(self,sizeInput) :
        # check if the size of the input is usable by the algorithm
        if sizeInput[0] < self.__kernelSize[0] or sizeInput[1] < self.__kernelSize[1] or sizeInput[2] != self.__kernelSize[2] :
            raise WrongDimention
        self.__sizeInput = sizeInput
    
    def getOutputSize(self) :
        return ((data.shape[0] + 2*self.__padding - kernel.shape[0])/self.__stride + 1,(data.shape[1] + 2*self.__padding - kernel.shape[1])/self.__stride + 1,kernel.shape[3])

    def compute(self,data,param) :
        # check if the kernel was given by the user or it have to be found
        if (self.__kernel == None).any() :
            # if the kernel have to be found, we buil it from 
            kernel = np.reshape(param, self.__kernelSize)
        else :
            # otherwise, we take the given kernel
            kernel = self.__kernel
        linearRes = np.zeros(((data.shape[0] + 2*self.__padding - kernel.shape[0])//self.__stride + 1,(data.shape[1] + 2*self.__padding - kernel.shape[1])//self.__stride + 1,kernel.shape[3]))
        # for each output channel, we're doing the 3D convolution
        for k in range(kernel.shape[3]) :
            # for each input channel, we're doing the 2D convolution and we add the results
            for c in range(kernel.shape[2]) :
                linearRes[:,:,k] = linearRes[:,:,k] + convolution2D(data[:,:,c],kernel[:,:,c,k],self.__stride,self.__padding)
        # we compute the result with the activation function before give it to the following layers
        return self.__activation(linearRes)

    def setKernel(self,kernel) :
        """fonction to set the kernel"""
        for i in range(len(kernel.shape)) :
            if kernel.shape[i] != self.__kernelSize[i] :
                raise WrongDimention
        for i in self.__kernelSize[len(kernel.shape):] :
            if i != 1 :
                raise WrongDimention
        if (len(kernel.shape) == 2) :
            self.__kernel = np.zeros((kernel.shape[0],kernel.shape[1],1,1))
            self.__kernel[:,:,0,0] = kernel
        elif (len(kernel.shape) == 3) :
            self.__kernel = np.zeros((kernel.shape[0],kernel.shape[1],kernel.shape[2],1))
            self.__kernel[:,:,:,0] = kernel
        elif (len(kernel.shape) == 4) :
            self.__kernel = kernel
        else :
            raise WrongDimention
        self.__nbParameter = 0
    
    def setStride(self,stride) :
        """fonction to modify the stride"""
        self.__stride = stride
    
    def setPadding(self,padding) :
        """fonction to modify the padding"""
        self.__padding = padding


class MaxPooling :
    """layer that perform a MaxPooling"""
    def __init__(self,sizePooling) :
        # check if the input is usable by the algorithm
        if (len(sizePooling) != 2) :
            raise WrongDimention
        # store the variable
        self.__size = sizePooling
        self.__stride = 1
        self.__padding = 0

    def getNbParameter(self) :
        # this layer doesn't take any parameter
        return 0
    
    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if ((inputSize[0] + 2*self.__padding - self.__size[0])%self.__stride != 0 or (inputSize[1] + 2*self.__padding - self.__size[1])%self.__stride != 0) :
            raise WrongDimention
        if (inputSize[0] < self.__size[0] or inputSize[1] < self.__size[1]) :
            raise WrongDimention
    
    def getOutputSize(self) :
        return ((data.shape[0] + 2*self.__padding - self.__size[0])//self.__stride + 1,(data.shape[1] + 2*self.__padding - self.__size[1])//self.__stride + 1,data.shape[2])
    
    def compute(self,data,param) :
        res = np.zeros(((data.shape[0] + 2*self.__padding - self.__size[0])//self.__stride + 1,(data.shape[1] + 2*self.__padding - self.__size[1])//self.__stride + 1,data.shape[2]))
        # padding the data
        if self.__padding != 0 :
            dataPadded = np.zeros((data.shape[0] + self.__padding*2, data.shape[1] + self.__padding*2))
            dataPadded[self.__padding:(-1 * self.__padding), self.__padding:(-1 * self.__padding)] = data
        else :
            dataPadded = data
        # k is the number of current channel we are working on
        for k in range(data.shape[2]) :
            # (i,j) is the coordinates of the extracted matrix from data we will perform the maximum
            for j in range(0, data.shape[1] - self.__size[1] + 1, self.__stride) :
                for i in range(0, data.shape[0] - self.__size[0] + 1, self.__stride) :
                    res[i//self.__stride, j//self.__stride, k] = data[i:i+self.__size[0],j:j+self.__size[1],k].max()
        return res
    
    def setStride(self,stride) :
        """fonction to modify the stride"""
        self.__stride = stride
    
    def setPadding(self,padding) :
        """fonction to modify the padding"""
        self.__padding = padding


class AveragePooling :
    """layer that perform an average pooling"""
    def __init__(self,sizePooling) :
        # check if the input is usable by the algorithm
        if (len(sizePooling) != 2) :
            raise WrongDimention
        # store the variable
        self.__size = sizePooling
        self.__stride = 1
        self.__padding = 0

    def getNbParameter(self) :
        # this layer doesn't take any parameter
        return 0

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if ((inputSize[0] + 2*self.__padding - self.__size[0])%self.__stride != 0 or (inputSize[1] + 2*self.__padding - self.__size[1])%self.__stride != 0) :
            raise WrongDimention
        if (inputSize[0] < self.__size[0] or inputSize[1] < self.__size[1]) :
            raise WrongDimention
    
    def getOutputSize(self) :
        return ((data.shape[0] + 2*self.__padding - self.__size[0])//self.__stride + 1,(data.shape[1] + 2*self.__padding - self.__size[1])//self.__stride + 1,data.shape[2])

    def compute(self,data,param) :
        res = np.zeros(((data.shape[0] + 2*self.__padding - self.__size[0])//self.__stride + 1,(data.shape[1] + 2*self.__padding - self.__size[1])//self.__stride + 1,data.shape[2]))
        # padding the data
        if self.__padding != 0 :
            dataPadded = np.zeros((data.shape[0] + self.__padding*2, data.shape[1] + self.__padding*2))
            dataPadded[self.__padding:(-1 * self.__padding), self.__padding:(-1 * self.__padding)] = data
        else :
            dataPadded = data
        # k is the number of current channel we are working on
        for k in range(data.shape[2]) :
            # (i,j) is the coordinates of the extracted matrix from data we will perform the average
            for j in range(0, data.shape[1] - self.__size[1] + 1, self.__stride) :
                for i in range(0, data.shape[0] - self.__size[0] + 1, self.__stride) :
                    res[i//self.__stride, j//self.__stride, k] = data[i:i+self.__size[0],j:j+self.__size[1],k].sum()/(float(self.__size[0]*self.__size[1]))
        return res
    
    def setStride(self,stride) :
        """fonction to modify the stride"""
        self.__stride = stride
    
    def setPadding(self,padding) :
        """fonction to modify the padding"""
        self.__padding = padding


class SimpleRNN :
    """a simple Recurent Neural Network layer"""
    def __init__(self,nbInternalUnits) :
        # number of internal units, and size of the output
        self.__nbInternalUnits = nbInternalUnits
        # set the defaut size of the output. The size of the second dimention will be added in SetInputSize
        self.__outputSize = (self.__nbInternalUnits,None)

    def getNbParameter(self) :
        # we have three square matrix of size (self.__inputSize[1],self.__inputSize[1])
        # and a vector of size self.__inputSize[1]
        return self.__inputSize[1] * (1 + 3 * self.__inputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize[1] = inputSize[1]

    def getOutputSize(self) :
        return self.__nbInternalUnits

    def compute(self,data,param) :
        innerState = param[0:self.__inputSize[1]]
        res = np.zeros(self.__nbInternalUnits,self.__inputSize[1])
        # I reshape the list of parameters in three square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        U = np.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        V = np.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        W = np.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1] + self.__outputSize[1])],(self.__outputSize[1],self.__inputSize[1]))
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = data[i,:]
            else :
                inputed = np.zeros(data.shape[1])
            # we update the inner state
            innerState = tanh( np.dot(U,inputed) + np.dot(V,innerState))
            # we compute the output with the new inner state
            res[i,:] = softmax(np.dot(W,innerState))
        return res[-self.__outputSize[0]:,:]

        def setOutputSize(self,outputSize) :
            if outputSize[0] > self.__innerState :
                raise WrongDimention
            self.__outputSize = ouputSize


class LSTM :
    """a Long Short Term Memory layer, an improved recurent neural network layer"""
    def __init__(self,nbInternalUnits) :
        # number of internal units, and size of the output
        self.__nbInternalUnits = nbInternalUnits
        # set the defaut size of the output. The size of the second dimention will be added in SetInputSize
        self.__outputSize = (self.__nbInternalUnits,None)

    def getNbParameter(self) :
        # we have nine square matrix of size (self.__inputSize[1],self.__inputSize[1])
        # a matrix of size (self.__outputSize[1],self.__inputSize[1])
        # and a vector of size self.__inputSize[1]
        return self.__inputSize[1] * (1 + 8 * self.__inputSize[1] + self.__outputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize[1] = inputSize[1]

    def getOutputSize(self) :
        return self.__nbInternalUnits

    def compute(self,data,param) :
        innerState = param[0:self.__inputSize[1]]
        res = np.zeros(self.__nbInternalUnits,self.__inputSize[1])
        # I reshape the list of parameters in nine square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        Ui = np.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vi = np.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Uf = np.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 3 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Vf = np.reshape(param[self.__inputSize[1] * (1 + 3*self.__inputSize[1]):self.__inputSize[1] * (1 + 4 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Uo = np.reshape(param[self.__inputSize[1] * (1 + 4*self.__inputSize[1]):self.__inputSize[1] * (1 + 5 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Vo = np.reshape(param[self.__inputSize[1] * (1 + 5*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        UnewMemory = np.reshape(param[self.__inputSize[1] * (1 + 6*self.__inputSize[1]):self.__inputSize[1] * (1 + 7 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        VnewMemory = np.reshape(param[self.__inputSize[1] * (1 + 7*self.__inputSize[1]):self.__inputSize[1] * (1 + 8 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        W = np.reshape(param[self.__inputSize[1] * (1 + 8*self.__inputSize[1]):self.__inputSize[1] * (1 + 8 * self.__inputSize[1] + self.__outputSize[1])](self.__outputSize[1],self.__inputSize[1]))
        # at the begining, there is nothing in the memory
        memory = np.zeros(self.__inputSize[1])
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = data[i,:]
            else :
                inputed = np.zeros(data.shape[1])
            # we update the inner state
            inputGate = sigmoid(np.dot(Ui,inputed) + np.dot(Vi,innerState))
            forgetGate = sigmoid(np.dot(Uf,inputed) + np.dot(Vf,innerState))
            ouputGate = sigmoid(np.dot(Uo,inputed) + np.dot(Vo,innerState))
            newMemory = tanh(np.dot(UnewMemory,inputed) + np.dot(VnewMemory,innerState))
            memory = inputgate * newMemory + forgetGate * memory
            innerState = ouputGate * tanh(memory)
            # we compute the output with the new inner state
            res[i,:] = softmax(np.dot(W,innerState))
        return res[-self.__outputSize[0]:,:]

        def setOutputSize(self,outputSize) :
            if outputSize[0] > self.__innerState :
                raise WrongDimention
            self.__outputSize = ouputSize

class GRU :
    """a Gated Recurent Unit layer, an improved recurent neural network layer"""
    def __init__(self,nbInternalUnits) :
        # number of internal units, and size of the output
        self.__nbInternalUnits = nbInternalUnits
        # set the defaut size of the output. The size of the second dimention will be added in SetInputSize
        self.__outputSize = (self.__nbInternalUnits,None)

    def getNbParameter(self) :
        # we have six square matrix of size (self.__inputSize[1],self.__inputSize[1]),
        # a matrix of size (self.__outputSize[1],self.__inputSize[1])
        # and a vector of size self.__inputSize[1]
        return self.__inputSize[1] * (1 + 6 * self.__inputSize[1] + outputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize[1] = inputSize[1]

    def getOutputSize(self) :
        return self.__nbInternalUnits

    def compute(self,data,param) :
        innerState = param[0:self.__inputSize[1]]
        res = np.zeros(seld.__nbInternalUnits,self.__outputSize[1])
        # I reshape the list of parameters in seven square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        Uz = np.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vz = np.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Ur = np.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 3 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Vr = np.reshape(param[self.__inputSize[1] * (1 + 3*self.__inputSize[1]):self.__inputSize[1] * (1 + 4 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Ug = np.reshape(param[self.__inputSize[1] * (1 + 4*self.__inputSize[1]):self.__inputSize[1] * (1 + 5 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        Vg = np.reshape(param[self.__inputSize[1] * (1 + 5*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1])](self.__inputSize[1],self.__inputSize[1]))
        W = np.reshape(param[self.__inputSize[1] * (1 + 6*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1] + self.__outputSize[1])](self.__outputSize[1],self.__inputSize[1]))
        # at the begining, there is nothing in the memory
        memory = np.zeros(self.__inputSize[1])
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = data[i,:]
            else :
                inputed = np.zeros(data.shape[1])
            # we update the inner state
            z = sigmoid(np.dot(Uz,inputed) + np.dot(Vz,innerState))
            r = sigmoid(np.dot(Ur,inputed) + np.dot(Vr,innerState))
            g = tanh(np.dot(Ug,inputed) + np.dot(Wg,r*innerState))
            innerState = z*g + (1 - f) * innerState
            # we compute the output with the new inner state
            res[i,:] = softmax(np.dot(W,innerState))
        return res[-self.__outputSize[0]:,:]
    
    def setOutputSize(self,outputSize) :
            if outputSize[0] > self.__innerState :
                raise WrongDimention
            self.__outputSize = ouputSize