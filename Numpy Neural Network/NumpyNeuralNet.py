#NumpyNeuralNet.py
#Author: Zach Stoebner
#Created on: 7-30-19
#Descrip: creating a neural net by hand using raw numpy
#Inspired by this article: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.openml import fetch_openml

#network class to call ops
class Network(object):
    
    #params will be defined as a list, otherwise manually enter args or dictionary
    def __init__(self,n_input,n_output,loss='mse',optimization='GradientDescent',learning_rate=0.1,threshold=0,**kwargs):
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
    def shuffle_batch(self,X, y, batch_size):
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
    def relu(self,X):
        return np.maximum(X,self.threshold)

    #mse loss function
    def mse(self,y_pred,y_true):
        return (1/y_true.shape[0]) * np.sum(np.square(y_pred - y_true))
    
    #SGD optimization function
    def GradientDescent(self,x,X,theta,y_pred,y_true,activation):
        if activation =='softmax':
            return (y_pred - y_true)
        ###FIGURE OUT GRADIENT DESCENT FOR SOFTMAX
        if activation == 'relu':
            r = self.relu(X)
            return np.dot(np.dot(X,(y_true - r)).T,x) * self.learning_rate #return delta-W
    
    
    #defining forward propagation for any one layer
    def forward_prop(self,X,theta,b,activation):
        if activation == 'relu':
            return self.relu(np.dot(X,theta) + b)
        
    #defining backprop
    def backward_prop(self,inputs,weights,y_pred,y_true):
        updated_weights = []
        if self.optimization == 'GradientDescent':
            for i in reversed(range(1,len(inputs))):
                delta = self.GradientDescent(inputs[i-1],inputs[i], weights[i-1], y_pred, y_true,self.activation[i-1])
                updated_weights.append(weights[i-1] - delta)
        return updated_weights.reverse()
            
    #fit data to model by training    
    def fit(self,X_in,y_in,
            epochs=100,batch_size=0.2):

        #training the network
        for epoch in range(epochs):
            outputs = np.empty(X_in.shape[0]) #matrix for this epoch's output
            for ind in range(len(X_in)): #running through each instance
                cur_inp = X_in[ind].copy()
                activities = [cur_inp,]
                for i in range(len(self.final_weights)): #forward prop through each layer
                    w = self.final_weights[i]
                    b = self.biases[i]
                    a = self.activation[i]
                    cur_inp = self.forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
                    activities.append(cur_inp) #saving activations at each layer, not including initial inputs
                outputs[ind] = np.argmax(activities[-1]) #discerning output from max activation at index
                
                #backprop algo
                self.final_weights = self.backward_prop(activities,self.final_weights,outputs,y_in).copy()
            
            cur_loss = self.mse(outputs,y_in) #loss for given epoch
            if epoch % 10 == 0:
                print('Epoch:',epoch,'\tLoss:',cur_loss)
        
        print("Training concluded")
    
    #generate predictions    
    def predict(self,X_in):
        outputs = np.empty(X_in.shape[0]) #matrix for this epoch's output
        for ind in range(X_in): #running through each instance
            cur_inp = X_in[ind].copy()
            for i in enumerate(self.final_weights): #forward prop through each layer
                w = self.final_weights[i]
                b = self.biases[i]
                a = self.activation[i]
                cur_inp = self.forward_prop(cur_inp.reshape(1,-1),w,b,a).copy()
            outputs[ind] = np.argmax(activities[-1]) #discerning output from max activation at index   
        return outputs
    
    def accuracy(self,y_pred,y_true):
        return np.sum((y_pred == y_true)) / y_true.shape[0]
        

#fetching mnist
X,y = fetch_openml('mnist_784',version=1,return_X_y=True)

#splitting data
test_size = 10000
X_train = X[:-2*test_size] #50,000 x 784
y_train = y[:-2*test_size]
X_val = X[-2*test_size:-test_size] #10,000 x 784
y_val = y[-2*test_size:-test_size]
X_test = X[-test_size:] #10,000 x 784
y_test = y[-test_size:]

n_inputs = X_train.shape[1]
n_outputs = np.unique(y).shape[0]
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
    
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.accuracy(y_pred, y_test))
    
    