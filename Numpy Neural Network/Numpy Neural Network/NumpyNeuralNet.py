#NumpyNeuralNet.py
#Author: Zach Stoebner
#Created on: 7-30-19
#Descrip: creating a neural net by hand using raw numpy
#Inspired by this article: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
#Copyright 2019. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Speed issue is fixed as long as batch_size is set.
-Need to revise algos because loss lodges at a solid 28.183 
-Need to implement early stopping
"""

####NETWORK CLASS
#initializes layers for a neural net
#Trains on inputted data
#Predicts output given new data
#Evaluates accuracy given predictions and true values
class Network(object):
    
    #params will be defined as a list, otherwise manually enter args or dictionary
    def __init__(self,n_input,n_output,loss='mse',optimization='SGD',learning_rate=0.01,threshold=0,**kwargs):
        self.n_input = n_input
        self.n_output = n_output
        self.loss = loss
        self.optimization = optimization
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.final_weights = []
        self.biases = []
        self.activation = []
        for key,value in kwargs.items():
            self.key = value
    
    #method to shuffle training sets into batches
    def __shuffle_batch(self,X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch
            
    #defining a layer constructor
    def layer(self,activation='relu',weights=None,biases=None):
        if weights.all() == None:
            weights = np.ones((self.n_input,self.n_output))
            biases = np.zeros((weights.shape[0],1))
            
        self.final_weights.append(weights)
        self.biases.append(biases)
        
        if activation == 'relu':
            self.activation.append('relu')
        if activation == 'softmax':
            self.activation.append('softmax')
        
    #relu activation function
    def __relu(self,X):
        return np.maximum(X,self.threshold)

    #mse loss function
    def __mse(self,y_pred,y_true):
        return (1/len(y_true)) * np.sum(np.square(y_pred - y_true))
    
    #SGD optimization function
    def __SGD(self,X,theta,y_pred,y_true):
        if self.loss == 'mse':
            return 2 * np.dot(X.T,X * (y_true - y_pred)) * self.learning_rate #return delta-W
    
    
    #defining forward propagation for any one layer
    def __forward_prop(self,X,theta,b,activation):
        if activation == 'relu':
            return self.__relu(np.dot(X,theta).T + b)
        
    #defining backprop
    def __backward_prop(self,inputs,weights,y_pred,y_true):
        updated_weights = []
        if self.optimization == 'SGD':
            for i in reversed(range(len(weights))):
                delta = self.__SGD(inputs[i], weights[i], y_pred, y_true)
                updated_weights.append((weights[i] - delta[0][0]*np.ones(weights[i].shape)))     
        return list(reversed(updated_weights))
            
    #fit data to model by training    
    def fit(self,X_in,y_in,
            epochs=100,batch_size=0.2,seed=42):

        np.random.seed(42)
        
        #training the network
        for epoch in range(epochs):
            if batch_size is None:
                outputs = np.empty(X_in.shape[0]) #matrix for this epoch's output
                for ind,cur_inp in enumerate(X_in): #running through each instance
                    activities = []
                    for i in range(len(self.final_weights)): #forward prop through each layer
                        w = self.final_weights[i]
                        b = self.biases[i]
                        a = self.activation[i]
                        cur_inp = self.__forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
                        activities.append(cur_inp) #saving activations at each layer, not including initial inputs
                    outputs[ind] = np.argmax(activities[-1]) #discerning output from max activation at index
            else:
                    
                #backprop algorithm given batch_size
                sample_y_pred,sample_y_true,sample_activities = self.__batch_predict(X_in, y_in,batch_size=batch_size)
                self.final_weights = self.__backward_prop(sample_activities,self.final_weights,sample_y_pred,sample_y_true)
                outputs = sample_y_pred
            
            cur_loss = self.__mse(outputs,y_in) #loss for given epoch
            if epoch % 10 == 0:
                print('Epoch:',epoch,'\tLoss:',cur_loss)
        
        print("Training concluded")
     
     #private helper method   
    def __batch_predict(self,X_in,y_in,batch_size=0.2):
        
        if self.optimization == 'SGD':
            rnd_ind = np.random.permutation(len(X_in))[0]
            X_batch,y_batch = np.reshape(X_in[rnd_ind],(-1,1)),y_in[rnd_ind]
            cur_inp = X_batch.copy()
            activities = []
            for i in range(len(self.final_weights)): #forward prop through each layer
                w = self.final_weights[i]
                b = self.biases[i]
                a = self.activation[i]
                cur_inp = self.__forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
                activities.append(cur_inp) #saving activations at each layer, not including initial inputs
            outputs = np.argmax(activities[-1]) #discerning output from max activation at index
            return outputs,np.int64(y_batch),activities             
            
        else:
            X_batch,y_batch = self.__shuffle_batch(X_in,y_in,batch_size)
            
        outputs = np.empty(X_batch.shape[0])
        for ind,cur_inp in enumerate(X_batch): #running through each instance
            activities = []
            for i in range(len(self.final_weights)): #forward prop through each layer
                w = self.final_weights[i]
                b = self.biases[i]
                a = self.activation[i]
                cur_inp = self.__forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
                activities.append(cur_inp) #saving activations at each layer, not including initial inputs
            outputs[ind] = np.argmax(activities[-1]) #discerning output from max activation at index
        return outputs,y_batch,activities      
    
    #generate predictions    
    def predict(self,X_in):
        outputs = np.empty(X_in.shape[0]) #matrix for this epoch's output
        for ind,cur_inp in enumerate(X_in): #running through each instance
            for i in range(len(self.final_weights)): #forward prop through each layer
                w = self.final_weights[i]
                b = self.biases[i]
                a = self.activation[i]
                cur_inp = self.__forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
            outputs[ind] = np.argmax(cur_inp)   
        return outputs
    
    def accuracy(self,y_pred,y_true):
        return np.sum((y_pred == y_true)) / len(y_true)


###SET-UP PHASE
#fetching mnist
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float64).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float64).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
    
n_inputs = X_train.shape[1]
n_outputs = np.unique(y_train).shape[0]
layer1_n_nodes = int(n_inputs/2)
layer2_n_nodes = int(np.floor(np.sqrt(n_inputs)))
    
W1 = np.random.randn(n_inputs,layer1_n_nodes)
W2 = np.random.randn(layer1_n_nodes,layer2_n_nodes)
W3 = np.random.randn(layer2_n_nodes,n_outputs)
b1 = np.zeros((layer1_n_nodes,1))
b2 = np.zeros((layer2_n_nodes,1))
b3 = np.zeros((n_outputs,1))
    
params = [W1, b1, W2, b2, W3, b3]
model = Network(n_inputs, n_outputs)
for pair in range(len(params) // 2):
    model.layer(weights=params[pair*2],biases=params[pair*2 + 1])

###EXECUTION PHASE    
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.accuracy(y_pred, y_test))
    