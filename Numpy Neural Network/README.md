# Numpy Neural Net
A raw numpy neural network

My intent is to continuously add functionality as long as I'm studying/working with ML. It's primarily for my own understanding. I'm only using NumPy to compute forward and backward propagation, no high-level APIs unless I'm importing a dataset.

I import the dataset using TensorFlow because 1. it loads faster than the SKLearn dataset and 2. SKlearn encodes the target set as strings (why THE FUCK does it code a written number as a string?)

THE MODEL RUNS!!!!!! But it has an exploding gradient issue and the algorithms are probably incorrect somewhere. The accuracy thus resolves to a measly 9.8%. 

# Current Progress:
 - Network constructor
 - Layer constructor
 - ReLU activation function
 - MSE function
 - Forward Propagation
 - Fit method == training
 - ReLU GD
 - Basic backprop
 - Predict method
 - Batch predict method
 - Accuracy method
 - Fixed slowdown and batch implementation
 
 # To-do:
 - Test on MNIST
 - Implement early stopping
 - Fix exploding gradient problem and overfitting
   - Batch normalization
   - Regularization
   - Xavier and He initialization
 - Implement visualization methods
 - Optimize CPU usage
 - Make it DRYer
 
