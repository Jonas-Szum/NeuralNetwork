#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import operator as operator
import numpy as np
np.random.seed(100)
#NOTE: input the training set (and correct outputs) as x and y, when calling NeuralNetwork()
#It automatically trains, so you don't need to call the train function

#from gradescope_utils.autograder_utils.decorators import weight, visibility
#import unittest
#from sklearn.model_selection import KFold
#import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import Normalizer
#import matplotlib.pyplot as mp


def sigmoid(t): #sigmoid function that will be used with the chain rule
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self,x = [[]],y = [],numLayers=2,numNodes=2,eta=5,maxIter=10000):
        self.input = np.array(x)
        
        self.y = []
        for i in range(len(y)):
            self.y.append( [(y[i])] )
        self.y = np.array(self.y)
        
        self.numLayers = numLayers
        self.numNodes = numNodes + 1 #make room for the bias node
        self.eta = eta
        self.maxIter = maxIter
        self.output = []
        self.weights = [np.random.uniform(-1,1,(len(x[0]), self.numNodes))] #create the weights from the inputs to the first layer
        
        for i in range(numLayers-1):
            self.weights.append(np.random.uniform(-1,1,(self.numNodes, self.numNodes))) #create the random weights between internal layers
        self.weights.append(np.random.uniform(-1,1,(self.numNodes, 1))) #create weights from final layer to output node
        self.train(learningRate=eta, maxIterations=maxIter) #immediately start training the NN

    def train(self, learningRate, maxIterations):
        for i in range(self.maxIter): #no early stopping, the loop always runs maxIter times
            self.feedforward()
            self.backprop()

    def predict(self,x=[]):
        self.input = []
        self.input.append(np.array(x))
        self.feedforward()
        print(self.output)
        for i in self.output:
            if(i[0] < .5):
                i[0] = 0 #predict a 1
            else:
                i[0] = 1 #predict a 5
        return self.output

    def feedforward(self): #prediction of value given the inputs
        self.layerData = [] 
        oneLayer = np.dot(self.input, self.weights[0]) #2D array
        self.layerData.append(sigmoid(oneLayer))
        for mat in self.layerData[0]:
            mat[self.numNodes-1] = 1.0 #set the bias node: 1
            #print(mat)
  
        #add layers in between input and output
        for i in range(1, self.numLayers):
            temp = np.dot(self.layerData[i-1], self.weights[i])
            self.layerData.append(sigmoid(temp))
            for mat in self.layerData[i]:
                mat[self.numNodes-1] = 1.0 #set the bias node: 1
        #add output layer
        temp = []
        temp = np.dot(self.layerData[self.numLayers - 1], self.weights[self.numLayers])
        self.layerData.append(sigmoid(temp))
        self.output = self.layerData[self.numLayers]
        
        
    def backprop(self): #the meat of the program, using gradient descent via the sigmoid function
        bpoutput = []  
        
        delta = (2*(self.y - self.output) * sigmoid_derivative(self.output))
        bpoutput.append(np.dot(self.layerData[self.numLayers-1].T, delta))
        dotWithCurrentLayer = []
        
        for i in range(self.numLayers-1, 0, -1):
            dotBPAndWeights = np.dot(delta, self.weights[i+1].T) #delta is the delta of the node 1 index higher
            delta = dotBPAndWeights * sigmoid_derivative(self.layerData[i])
            bpoutput.append(np.dot(self.layerData[i].T, delta))
        
        dotFirst = np.dot(delta, self.weights[1].T)
        delta = dotFirst * sigmoid_derivative(self.layerData[0])
        bpoutput.append(np.dot(self.input.T, delta))
                
        x = self.numLayers
        for mat in self.weights:
            mat += self.eta*bpoutput[x]
            x = x - 1


# In[ ]:




