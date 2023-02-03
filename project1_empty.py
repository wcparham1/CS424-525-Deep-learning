import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear represented as: f(x) = x
1 - logistic (only one supported) represented as: 1/(1+np.exp(-x))
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    
    #store input, output, and partial derivative for each weight
    n_inputs = []
    n_outputs = []
    partial_dervs = []    
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        
        #if no weights are specified set them to arbitrary/random values
        if(weights == None):
            self.weights = [1] * self.input_num
            
            #initialize random weight values
            for i in range(0, len(self.weights)):
                self.weights[i] = np.random.randint(1,9)
        else:
            self.weights = weights
  
        #append 1 to represent bias
        self.weights.append(1)
    
        #print(self.activation, ' ', self.input_num, ' ', self.lr, ' ', self.weights)
        
        print('constructor')    
        
    #This method returns the activation of the net (net same as input for neuron)
    def activate(self,net):
        
        #first check for which activation function the neuron is using
        if (self.activation == 0):
            return net
        
        elif (self.activation == 1):
            return (1/(1+np.exp(-net)))
        print('activate')   
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        print('calculate')

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        print('calcpartialderivative') 
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        print('updateweight')

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        print('constructor') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        print('calculate') 
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        print('constructor') 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        print('constructor')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        print('calculate')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
        ner1 = Neuron(0, 3, 0.3)
        
        
        
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1]) #changed from x== to x=
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')