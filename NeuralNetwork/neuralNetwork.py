import math
import numpy as np
import tensorflow as tf

# Exceptions used
class UnknowFunction(Exception) :
    pass

class WrongDimention(Exception) :
    pass

class UnusableType(Exception) :
    pass


# activation functions
def relu(z) :
    return tf.nn.relu(z)
    
def sigmoid(z) :
    return tf.math.sigmoid(z)

def tanh(z) :
    return tf.math.tanhs(z)

def softmax(z) :
    return tf.math.softmax(z)


# utility functions
def convolution2D(data,kernel,stride=1,padding=0) :
    """perform a 2D convolution between a matrix and a kernel"""
    lenKernel = kernel.shape[0] 
    widKernel = kernel.shape[1] 
    lenData = data.shape[0] 
    widData = data.shape[1]
    lenRes = int(((lenData - lenKernel + 2 * padding) // stride) + 1)
    widRes = int(((widData - widKernel + 2 * padding) // stride) + 1)
    res = []
    if padding != 0 :
        dataPadded = tf.concat([tf.zeros((padding,widData),"double"),data,tf.zeros((padding,widData),"double")],0)
        dataPadded = tf.concat([tf.zeros((lenData + 2*padding,padding),"double"),dataPadded,tf.zeros((lenData + 2*padding,padding),"double")],1)
    else :
        dataPadded = data
    for j in range(0, dataPadded.shape[1] - widKernel + 1, stride) :
        for i in range(0, dataPadded.shape[0] - lenKernel + 1, stride) :
            # Tensor doesn't accept term assignment so I have to do a list of the term and concatenante and reshape it for the same operation
            res.append(tf.reshape(tf.reduce_sum((tf.math.multiply(kernel, dataPadded[i: i + lenKernel, j: j + widKernel]))),(1,1)))
    return tf.transpose(tf.reshape(tf.concat(res,0),(widRes,lenRes)))

def checkActivation(name) :
    """check if the name corespond to an existing activation function and return the right function"""
    if name == "relu" :
        return lambda z : relu(z)
    elif name == "sigmoid" :
        return lambda z : sigmoid(z)
    elif name == "tanh" :
        return lambda z : tanh(z)
    elif name == "softmax" :
        return lambda z : softmax(z)
    else :
        raise UnknowFunction(name)


# default score
def noScore(restheo, rescomputed) :
    """the score used if no custom score has been inputed by the user"""
    return tf.math.reduce_sum(tf.math.abs(tf.constant(restheo) - rescomputed))


# definition of class to create a neural network
class Network :
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
        self.__nbParameter = nbParameter
        # optimisation of the parameters
        tf.compat.v1.enable_eager_execution()
        params = tf.Variable(np.random.randn(nbParameter))
        loss = lambda : self.loss(data,result,score,params)
        opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
        opt.minimize(loss,[params])
        self.__params = params

    def compute(self,data,iflearning=False) :
        """compute the use of the neural network after the learning"""
        tf.compat.v1.enable_eager_execution()
        # first, we check if the dimention of the data is right
        if (data.shape != self.__inputSize) :
            raise WrongDimention
        # for each layer, we give the inner data between the previous layer and the one we work on it
        # and the list of parameters of the neural network, amputed from their first parameters, in a
        # way that the first parameters of the given list is the parameters of the layer
        dataTF = tf.constant(data)
        index = 0
        for layer in self.__layers :
            dataTF = layer.compute(dataTF,self.__params[index:index+layer.getNbParameter()])
            index = index + layer.getNbParameter()
        # if the program is learning, compute return a tensor. If the user launch it, it return a numpy array, with same information as the tensor
        if iflearning :
            return dataTF
        else :
            return dataTF.numpy()
    
    def loss(self,data,result,score,params) :
        """describe the loss needed for the optimisation"""
        # calcul the mean of the score of each computation of one data and the score needed
        self.__params = params
        res = tf.constant(0,dtype="double")
        n = len(result)
        if len(data.shape) == 2 :
            for i in range(n) :
                res = res + score(result[i,:], self.compute(data[i,:],iflearning=True))/n
        elif len(data.shape) == 3 :
            for i in range(n) :
                res = res + score(result[i,:], self.compute(data[i,:,:],iflearning=True))/n
        elif len(data.shape) == 4 :
            for i in range(n) :
                res = res + score(result[i,:], self.compute(data[i,:,:,:],iflearning=True))/n
        else :
            raise WrongDimentions
        return res
    
    def saveParameters(self, fileName) :
        # save the parameters on a file, in order to load it from an another session
        file = open(fileName,"w")
        string = str(self.__params[0])
        for i in range(1,self.__nbParameter) :
            string = string + "\n" + str(self.__params[i])
        file.write(string)
        file.close()
    
    def loadParameters(self, fileName) :
        # load the parameters on a file, save from an another session
        file = open(fileName,"r")
        string = file.read()
        file.close()
        paramsStr = string.split("\n")
        paramsFloat
        for param in params :
            paramsFloat = paramsFloat + [float(param)]
        self.__params = tf.Variable(paramsFloat,dtype="double")



# class that define a layer. Each class must have the getNbParameter() and compute(data,param) functions
class DenseLayer :
    """a class representing a dense layer in a neural network"""
    def __init__(self,lengthOutput,activation) :
        # size of the output of the layer, given by the user
        if not(isinstance(lengthOutput,int)) :
            raise UnusableType
        self.__sizeOutput = lengthOutput
        # chose what activation function the user want to use
        self.__activation = checkActivation(activation)
    
    def getNbParameter(self) :
        # just one matrix of size (self.__sizeNeuron[0], self.__sizeNeuron[1])
        return self.__sizeNeuron[0] * self.__sizeNeuron[1]
    
    def setInputSize(self,inputSize) :
        # compute the number of input
        if not(isinstance(inputSize,int)) :
            raise WrongDimention
        # the size of the matrix representing the layer
        self.__sizeNeuron = (self.__sizeOutput,inputSize)
    
    def getOutputSize(self) :
        return self.__sizeOutput
    
    def compute(self,data,param) :
        # construct the matrix representing the layer from the list of parameters
        neuron = tf.reshape(param,self.__sizeNeuron)
        # compute the result before the activation
        linearResult = tf.reshape(tf.linalg.matmul(neuron,tf.reshape(data,[-1,1])),[-1])
        # compute the result and return it after the activation
        return self.__activation(linearResult)
    

class FlattenLayer :
    """layer that vectorize the data"""
    def __init(self) :
        # there is nothing to initialize
        pass

    def getNbParameter(self) :
        # this layer doesn't take any parameter
        return 0
    
    def setInputSize(self, inputSize) :
        self.__inputSize = inputSize
    
    def getOutputSize(self) :
        res = 1
        for i in self.__inputSize :
            res = res*i
        return res
    
    def compute(self,data,param) :
        return tf.reshape(data,[-1])


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
    
    def compute(self,data,param) :
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
        self.__activation = checkActivation(activation)
        # calculate the number of parameters needed
        self.__nbParameter = 1
        for dimension in kernelSize :
            self.__nbParameter = self.__nbParameter * dimension
        # the kernel used in the convolution
        self.__kernelInputed = False
        # the parameter of the convolution
        self.__stride = 1
        self.__padding = 0
    
    def getNbParameter(self) :
        return self.__nbParameter
    
    def setInputSize(self,sizeInput) :
        # check if the size of the input is usable by the algorithm
        if sizeInput[1] < self.__kernelSize[2] or sizeInput[2] < self.__kernelSize[3] or sizeInput[0] != self.__kernelSize[1] :
            raise WrongDimention
        self.__inputSize = sizeInput
    
    def getOutputSize(self) :
        return (self.__kernelSize[0],(self.__inputSize[1] + 2*self.__padding - self.__kernelSize[2])/self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__kernelSize[3])/self.__stride + 1)

    def compute(self,data,param) :
        # check if the kernel was given by the user or it have to be found
        if not(self.__kernelInputed) :
            # if the kernel have to be found, we buil it from 
            kernel = tf.reshape(param, self.__kernelSize)
        else :
            # otherwise, we take the given kernel
            kernel = self.__kernel
        linearRes = []
        # for each output channel, we're doing the 3D convolution
        for k in range(kernel.shape[0]) :
            # for each input channel, we're doing the 2D convolution and we add the results
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            linearResChannel = tf.zeros((1,(self.__inputSize[1] + 2*self.__padding - self.__kernelSize[2])/self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__kernelSize[3])/self.__stride + 1),"double")
            for c in range(kernel.shape[1]) :
                linearResChannel = linearResChannel + tf.reshape(convolution2D(data[c,:,:],kernel[k,c,:,:],self.__stride,self.__padding),(1,(self.__inputSize[1] + 2*self.__padding - self.__kernelSize[2])//self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__kernelSize[3])//self.__stride + 1))
            linearRes.append(linearResChannel)
        # we compute the result with the activation function before give it to the following layers
        return self.__activation(tf.concat(linearRes,0))

    def setKernel(self,kernel) :
        """fonction to set the kernel"""
        # check if the kernel has the right dimention
        j = 0
        for i in range(4-len(kernel.shape),4) :
            if kernel.shape[j] != self.__kernelSize[i] :
                raise WrongDimention
            j = j+1
        for i in self.__kernelSize[:4-len(kernel.shape)] :
            if i != 1 :
                raise WrongDimention
        # transform the inputed kernel into a 4 dimention tensor
        if (len(kernel.shape) == 2) :
            self.__kernel = np.zeros((1,1,kernel.shape[0],kernel.shape[1]))
            self.__kernel[0,0:,:] = kernel
            self.__kernel = tf.constant(self.__kernel,dtype="double")
        elif (len(kernel.shape) == 3) :
            self.__kernel = np.zeros((1,kernel.shape[0],kernel.shape[1],kernel.shape[2]))
            self.__kernel[0,:,:,:] = kernel
            self.__kernel = tf.constant(self.__kernel,dtype="double")
        elif (len(kernel.shape) == 4) :
            self.__kernel = tf.constant(kernel,dtype="double")
        else :
            raise WrongDimention
        self.__nbParameter = 0
        self.__kernelInputed = True
    
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
        if ((inputSize[1] + 2*self.__padding - self.__size[0])%self.__stride != 0 or (inputSize[2] + 2*self.__padding - self.__size[1])%self.__stride != 0) :
            raise WrongDimention
        if (inputSize[1] < self.__size[0] or inputSize[2] < self.__size[1]) :
            raise WrongDimention
        self.__inputSize = inputSize
    
    def getOutputSize(self) :
        return (self.__inputSize[0],(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1)
    
    def compute(self,data,param) :
        res = []
        # padding the data
        if self.__padding != 0 :
            dataPadded = tf.concat([tf.zeros((padding,widData),"double"),data,tf.zeros((padding,widData),"double")],0)
            dataPadded = tf.concat([tf.zeros((lenData + 2*padding,padding),"double"),dataPadded,tf.zeros((lenData + 2*padding,padding),"double")],1)
        else :
            dataPadded = data
        # k is the number of current channel we are working on
        for k in range(data.shape[0]) :
            # (i,j) is the coordinates of the extracted matrix from data we will perform the maximum
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            resChannel = []
            for j in range(0, data.shape[2] - self.__size[1] + 1, self.__stride) :
                for i in range(0, data.shape[1] - self.__size[0] + 1, self.__stride) :
                    resChannel.append(tf.reshape(tf.reduce_max(dataPadded[k,i:i+self.__size[0],j:j+self.__size[1]]),(1,1)))
            res.append(tf.reshape(tf.transpose(tf.reshape(tf.concat(resChannel,0),((self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1,(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1))),(1,(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1)))
        return tf.concat(res,0)
    
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
        if ((inputSize[1] + 2*self.__padding - self.__size[0])%self.__stride != 0 or (inputSize[2] + 2*self.__padding - self.__size[1])%self.__stride != 0) :
            raise WrongDimention
        if (inputSize[1] < self.__size[0] or inputSize[2] < self.__size[1]) :
            raise WrongDimention
        self.__inputSize = inputSize
    
    def getOutputSize(self) :
        return (self.__inputSize[0],(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1)

    def compute(self,data,param) :
        res = []
        # padding the data
        if self.__padding != 0 :
            dataPadded = tf.concat([tf.zeros((padding,widData),"double"),data,tf.zeros((padding,widData),"double")],0)
            dataPadded = tf.concat([tf.zeros((lenData + 2*padding,padding),"double"),dataPadded,tf.zeros((lenData + 2*padding,padding),"double")],1)
        else :
            dataPadded = data
        # k is the number of current channel we are working on
        for k in range(data.shape[0]) :
            # (i,j) is the coordinates of the extracted matrix from data we will perform the maximum
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            resChannel = []
            for j in range(0, data.shape[2] - self.__size[1] + 1, self.__stride) :
                for i in range(0, data.shape[1] - self.__size[0] + 1, self.__stride) :
                    resChannel.append(tf.reshape(tf.reduce_mean(dataPadded[k,i:i+self.__size[0],j:j+self.__size[1]]),(1,1)))
            res.append(tf.reshape(tf.transpose(tf.reshape(tf.concat(resChannel,0),((self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1,(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1))),(1,(self.__inputSize[1] + 2*self.__padding - self.__size[0])//self.__stride + 1,(self.__inputSize[2] + 2*self.__padding - self.__size[1])//self.__stride + 1)))
        return tf.concat(res,0)
    
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
        # we have two square matrix of size (self.__inputSize[1],self.__inputSize[1]),
        # one matrix of size (self.__outputSize[1],self.__inputSize[1])
        # and a vector of size self.__inputSize[1]
        return self.__inputSize[1] * (1 + 2 * self.__inputSize[1] + self.__outputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize[1] > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize = (self.__outputSize[0],inputSize[1])

    def getOutputSize(self) :
        return self.__outputSize

    def compute(self,data,param) :
        innerState = tf.reshape(param[0:self.__inputSize[1]],(-1,1))
        res = []
        # I reshape the list of parameters in three square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        U = tf.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        V = tf.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        W = tf.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1] + self.__outputSize[1])],(self.__outputSize[1],self.__inputSize[1]))
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = tf.reshape((data[i,:]),(-1,1))
            else :
                inputed = tf.zeros((self.__inputSize[1],1))
            # we update the inner state
            innerState = tf.math.tanh( tf.matmul(U,inputed) + tf.matmul(V,innerState))
            # we compute the output with the new inner state
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            res.append(tf.reshape(tf.math.softmax(np.dot(W,innerState)),(1,-1)))
        returned = tf.reshape(tf.concat(res,1),(self.__nbInternalUnits,self.__outputSize[1]))
        return returned[-self.__outputSize[0]:,:]

    def setOutputSize(self,outputSize) :
        if outputSize[0] > self.__nbInternalUnits :
            raise WrongDimention
        self.__outputSize = outputSize


class LSTM :
    """a Long Short Term Memory layer, an improved recurent neural network layer"""
    def __init__(self,nbInternalUnits) :
        # number of internal units, and size of the output
        self.__nbInternalUnits = nbInternalUnits
        # set the defaut size of the output. The size of the second dimention will be added in SetInputSize
        self.__outputSize = (self.__nbInternalUnits,None)

    def getNbParameter(self) :
        # we have eight square matrix of size (self.__inputSize[1],self.__inputSize[1])
        # a matrix of size (self.__outputSize[1],self.__inputSize[1])
        # and a vector of size self.__inputSize[1]
        return self.__inputSize[1] * (1 + 8 * self.__inputSize[1] + self.__outputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize[1] > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize = (self.__outputSize[0],inputSize[1])

    def getOutputSize(self) :
        return self.__outputSize

    def compute(self,data,param) :
        innerState = tf.reshape(param[0:self.__inputSize[1]],(-1,1))
        res = []
        # I reshape the list of parameters in nine square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        Ui = tf.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vi = tf.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Uf = tf.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 3 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vf = tf.reshape(param[self.__inputSize[1] * (1 + 3*self.__inputSize[1]):self.__inputSize[1] * (1 + 4 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Uo = tf.reshape(param[self.__inputSize[1] * (1 + 4*self.__inputSize[1]):self.__inputSize[1] * (1 + 5 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vo = tf.reshape(param[self.__inputSize[1] * (1 + 5*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        UnewMemory = tf.reshape(param[self.__inputSize[1] * (1 + 6*self.__inputSize[1]):self.__inputSize[1] * (1 + 7 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        VnewMemory = tf.reshape(param[self.__inputSize[1] * (1 + 7*self.__inputSize[1]):self.__inputSize[1] * (1 + 8 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        W = tf.reshape(param[self.__inputSize[1] * (1 + 8*self.__inputSize[1]):self.__inputSize[1] * (1 + 8 * self.__inputSize[1] + self.__outputSize[1])],(self.__outputSize[1],self.__inputSize[1]))
        # at the begining, there is nothing in the memory
        memory = tf.zeros((self.__inputSize[1],1))
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = tf.reshape((data[i,:]),(-1,1))
            else :
                inputed = tf.zeros((self.__inputSize[1],1))
            # we update the inner state
            inputGate = tf.math.sigmoid(tf.matmul(Ui,inputed) + tf.matmul(Vi,innerState))
            forgetGate = tf.math.sigmoid(tf.matmul(Uf,inputed) + tf.matmul(Vf,innerState))
            outputGate = tf.math.sigmoid(tf.matmul(Uo,inputed) + tf.matmul(Vo,innerState))
            newMemory = tf.math.tanh(tf.matmul(UnewMemory,inputed) + tf.matmul(VnewMemory,innerState))
            memory = tf.math.multiply(inputGate, newMemory) + tf.math.multiply(forgetGate, memory)
            innerState = tf.math.multiply(outputGate, tf.tanh(memory))
            # we compute the output with the new inner state
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            res.append(tf.reshape(tf.math.softmax(np.dot(W,innerState)),(1,-1)))
        returned = tf.reshape(tf.concat(res,1),(self.__nbInternalUnits,self.__outputSize[1]))
        return returned[-self.__outputSize[0]:,:]

    def setOutputSize(self,outputSize) :
        if outputSize[0] > self.__nbInternalUnits :
            raise WrongDimention
        self.__outputSize = outputSize

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
        return self.__inputSize[1] * (1 + 6 * self.__inputSize[1] + self.__outputSize[1])

    def setInputSize(self, inputSize) :
        # check if the input is usable by the algorithm
        if inputSize[1] > self.__nbInternalUnits :
            raise WrongDimention
        self.__inputSize = inputSize
        # if there is no size for the second dimention of the input, set a defaut one
        if self.__outputSize[1] == None :
            self.__outputSize = (self.__outputSize[0],inputSize[1])

    def getOutputSize(self) :
        return self.__outputSize

    def compute(self,data,param) :
        innerState = tf.reshape(param[0:self.__inputSize[1]],(-1,1))
        res = []
        # I reshape the list of parameters in seven square matrix. I arbitrarily choose here to set the inner dimention and the output dimention to the same size
        # of each data
        Uz = tf.reshape(param[self.__inputSize[1]:self.__inputSize[1] * (1 + self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vz = tf.reshape(param[self.__inputSize[1] * (1 + self.__inputSize[1]):self.__inputSize[1] * (1 + 2 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Ur = tf.reshape(param[self.__inputSize[1] * (1 + 2*self.__inputSize[1]):self.__inputSize[1] * (1 + 3 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vr = tf.reshape(param[self.__inputSize[1] * (1 + 3*self.__inputSize[1]):self.__inputSize[1] * (1 + 4 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Ug = tf.reshape(param[self.__inputSize[1] * (1 + 4*self.__inputSize[1]):self.__inputSize[1] * (1 + 5 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        Vg = tf.reshape(param[self.__inputSize[1] * (1 + 5*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1])],(self.__inputSize[1],self.__inputSize[1]))
        W = tf.reshape(param[self.__inputSize[1] * (1 + 6*self.__inputSize[1]):self.__inputSize[1] * (1 + 6 * self.__inputSize[1] + self.__outputSize[1])],(self.__outputSize[1],self.__inputSize[1]))
        # at the begining, there is nothing in the memory
        memory = tf.zeros(self.__inputSize[1])
        for i in range(self.__nbInternalUnits) :
            # we chose if there is an input or not
            if i<data.shape[0] :
                inputed = tf.reshape((data[i,:]),(-1,1))
            else :
                inputed = tf.zeros((self.__inputSize[1],1))
            # we update the inner state
            z = tf.math.sigmoid(tf.matmul(Uz,inputed) + tf.matmul(Vz,innerState))
            r = tf.math.sigmoid(tf.matmul(Ur,inputed) + tf.matmul(Vr,innerState))
            g = tf.math.tanh(tf.matmul(Ug,inputed) + tf.matmul(Vg,tf.math.multiply(r,innerState)))
            innerState = tf.math.multiply(z,g) + tf.multiply((1 - z), innerState)
            # we compute the output with the new inner state
            # tensor doesn't accept term assignment, so I have to concatenate and reshape the result in order to do a term assignment
            res.append(tf.reshape(tf.math.softmax(np.dot(W,innerState)),(1,-1)))
        returned = tf.reshape(tf.concat(res,1),(self.__nbInternalUnits,self.__outputSize[1]))
        return returned[-self.__outputSize[0]:,:]
    
    def setOutputSize(self,outputSize) :
            if outputSize[0] > self.__nbInternalUnits :
                raise WrongDimention
            self.__outputSize = outputSize
