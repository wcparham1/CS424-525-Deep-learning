#Authors: William Connor Parham, Coby White
#Due Date: February 13, 2023
#Professor: Amir Sadovnik
#Class: Computer Science 424/525 Deep learning
#Description: This lab is an implementation of Deep Neural Networks using object oriented Python.


import numpy as np
import matplotlib.pyplot as plt
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
    n_input = []
    n_output = 0
    partial_dervs = []    
    
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        
        #if no weights are specified set them to arbitrary/random values
        if(weights is None):
            self.weights = [1] * (self.input_num + 1)  #add 1 here to represent the bias input
            
            #initialize random weight values
            for i in range(0, len(self.weights)):
                self.weights[i] = np.random.randint(1,9)
        else:
            self.weights = weights

        
    #This method returns the activation of the net (net same as input for neuron)
    def activate(self,net):
        if (self.activation == 0):
            #save output for backpropagation
            self.output = net
            return net
        
        elif (self.activation == 1):
            #save output for backpropagation
            self.output = (1/(1+np.exp(-net)))
            return 1/(1+np.exp(-net))
        
    #Calculate the output of the neuron should save the input and output for back-propagation.  --Calculate calls activate-- 
    def calculate(self,input):
        #each input will correspond with each weight in the same place in the weights array
        #find the sum of the inputs * weights
        net = 0        
        sum = 0

        #sum up input to neuron
        for i in range(0, (len(input))):
            sum = input[i] * self.weights[i]
            net += sum
        
        #save input for backpropagation
        self.n_input = net
        
        output = self.activate(net)
        self.n_output = output
        return output
     
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if(self.activation == 0):
            #logistic activation returns 1
            return (1)
        elif(self.activation == 1):
            #return (np.exp(self.n_input))/((np.exp(self.n_input)+1)**2)
            return self.output * (1 - self.output)
        #print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        #find partial derivative
        self.partial_derivative = self.activationderivative() * wtimesdelta * self.n_input
        
        self.partial_dervs = np.dot(self.partial_derivative, self.weights)

        return self.partial_dervs
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        #simply update our weights
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] - (self.lr * self.partial_dervs[i])
        
    #helper print function
    def print_info(self):
        print('Neuron info: Activation: ', self.activation, ' input_num: ', self.input_num, ' learning rate: ', self.lr, ' weights: ', self.weights)

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.neuron_list = []
        
        #initialize random weights if ungiven
        if(weights is None):
            self.weights = np.random.randint(1, 9, [self.input_num, self.numOfNeurons]) 
        else:
            self.weights = weights
        
        #create our list of neurons for operations
        for i in range(0, self.numOfNeurons):
            new_neuron = Neuron(self.activation, self.input_num, self.lr, self.weights[i])
            self.neuron_list.append(new_neuron)
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        neuron_calculations = []
        
        #perform neuron activation calculations
        for i in range (0, len(self.neuron_list)):
            temp = self.neuron_list[i].calculate(input)
            neuron_calculations.append(temp)
    
        return neuron_calculations
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        
        #calculate the next layers wdelta and iterate
        wtimesdelta = np.array(wtimesdelta).flatten()
        ret = np.zeros((self.numOfNeurons, (len(self.weights)+1)))
        
        for i, neuron in enumerate(self.neuron_list):
            neuron_delta = neuron.calcpartialderivative(wtimesdelta[i])
            ret[i,:] = neuron_delta
            neuron.updateweight()
        
        #return the sum
        return np.sum(ret, axis=0)
                 
    #helper function to print information
    def print_info(self):
        print("fully_connected: num neurons: ", self.numOfNeurons, ' activation: ', self.activation, ' input_num: ', self.input_num, ' lr: ', self.lr, ' weights: ', self.weights)
           
           
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.iteration = []
        self.train_count = 0
        self.numOfLayers = numOfLayers
        self.numOfNeurons = list(numOfNeurons)
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.layer_list = []
        self.loss_list = []
        
        #initialize random weights if none are given
        if(weights is None):
            self.weights = []
            llsize = self.inputSize
            for i in range(self.numOfLayers):
                self.weights.append([])
                for j in range(self.numOfNeurons[i]):
                    self.weights[i].append([])
                    for k in range(llsize):
                        self.weights[i][j].append(np.random.random())
                llsize = self.numOfNeurons[i]
        else:
            self.weights = weights

        #create networks according to specification
        for i in range(0, self.numOfLayers):
            if(self.numOfLayers <= 1):
                new_layer = FullyConnected(self.numOfNeurons[i], self.activation, self.inputSize, self.lr, self.weights)

            else:
                new_layer = FullyConnected(self.numOfNeurons[i], self.activation, self.inputSize, self.lr, self.weights[i]) 
            
            self.layer_list.append(new_layer)
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,x):
        #perform large calculation
        x = np.array(x)
        current_input = x
        for i in range(0, len(self.layer_list)):
            current_input = np.concatenate((current_input, [1.0]))
            current_input = self.layer_list[i].calculate(current_input)
        self.y_hat = current_input
        return current_input
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # MSE
        #self.loss = np.mean(np.square(yp - y))
        if(self.loss == 0):
            return np.mean(np.square(yp - y))
        elif(self.loss == 1):
            t0 = (1-y) * np.log(1-yp + 1e-7)
            t1 = y * np.log(yp + 1e-7)
            return -np.mean(t0+t1, axis=0)
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        return -(y-yp)
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        self.iteration.append(self.train_count)
        out = self.calculate(x) #performing feed forward pass
        self.loss_list.append(self.calculateloss(self.y_hat, y)) #find derivative of the loss function
        delta = np.empty((1,self.numOfNeurons[-1]))
        delta = delta.flatten()
        
        for i in range(self.numOfNeurons[-1]):
            delta[i] = self.lossderiv(self.y_hat[i], y[i])

        for j in reversed(range(len(self.layer_list))):
            delta = self.layer_list[j].calcwdeltas(delta)
        self.train_count += 1
    
    def print_info(self):
        print(self.numOfLayers,' ',self.numOfNeurons,' ',self.inputSize,' ',self.activation,' ',self.loss,' ',self.lr)



###############        NEED TO GRAPH AND ADD SECOND LOSS FUNCTION        ##############
if __name__=="__main__":
    if (len(sys.argv)<2):
        print('usage: python project1_finished.py [learning rate] ["and" , "xor" , "example"]\n If no learning rate is specified it will be set to 0.5 and run the "example" option.')
        #network parameters:
            #self -- omitted 1
            #numOfLayers     2
            #numOfNeurons    3
            #inputSize       4
            #activation      5
            #loss            6
            #lr              7
            #weights         8
        lr = 0.5
        x = [[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]] 
        #                          2 3               4 5 6 7   8
        
        neural_net = NeuralNetwork(2,np.array([2,2]),2,1,0,lr,x)
        ret = neural_net.calculate([.05,.10])
        neural_net.train(np.array([0.05, .10]), np.array([.01, .99]))
        for i in range(0,10):
             neural_net.train(np.array([0.05, .10]), np.array([.01, .99]))
        
        #####################################################################
        
        neural_net2 = NeuralNetwork(2,np.array([2,2]),2,1,0,0.1,x)
        ret2 = neural_net2.calculate([.05,.10])
        neural_net2.train(np.array([0.05, .10]), np.array([.01, .99]))
        for i in range(0,10):
            neural_net2.train(np.array([0.05, .10]), np.array([.01, .99]))
        
         
        #print(neural_net.loss_list, '\n')
        print(neural_net2.loss_list)
        #print('ret:', ret)
        
        plt.scatter(neural_net2.loss_list, neural_net2.iteration, color='orange', label='0.1 lr')
        plt.scatter(neural_net.loss_list, neural_net.iteration, color='blue', label='0.5 lr')
        plt.xticks([0.22,0.24,0.26,0.28,0.30], ['0.30','0.28','0.26','0.24','0.22'])
        plt.yticks([0,2,4,6,8,10], ['10','8','6','4','2','0'])
        plt.xlabel('Loss')
        plt.ylabel('Training Iteration')
        plt.title("Loss Vs. Training Iteration")
        plt.legend()
    
        plt.show()
        
    elif (sys.argv[2]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]) #<- this is weight
        x=np.array([0.05,0.1])                                              #<- this is input
        e_o = np.array([0.01,0.99])                                         #<- this is expected output
        lr = sys.argv[1]
        lr = float(lr)
        #                          2 3               4 5 6 7   8
        neural_net = NeuralNetwork(2,np.array([2,2]),2,1,0,lr,w)
        ret = neural_net.calculate(x)
        
        print('weights before training:\n', neural_net.weights)
        neural_net.train(x,e_o)
        print('weights after training:\n', neural_net.weights)

        print('output: ',ret)
        
    elif(sys.argv[2]=='and'):
        x = np.array([1,0])  #<- this is input

        e_o = np.array([0])    #<- this is expected output
        lr = sys.argv[1]
        lr = float(lr)         #<- this is learning rate
        w = np.array([[.5,.5,.5]])  #<- this is weight
        #network parameters:
            #self -- omitted 1
            #numOfLayers     2
            #numOfNeurons    3
            #inputSize       4
            #activation      5
            #loss            6
            #lr              7
            #weights         8
        #                          2, 3              4  5  6  7   8
        neural_net = NeuralNetwork(1, np.array([1]), 2, 1, 0, lr, w)
        ret = neural_net.calculate(x)
        print('ret:', ret)
        #neural_net.train(x, e_o)
        print('learn and')
        
    elif(sys.argv[2]=='xor'):
        x = np.array([1,0])  #<- this is input

        e_o = np.array([1])    #<- this is expected output
        lr = sys.argv[1]
        lr = float(lr)         #<- this is learning rate
        w = np.array([[.5,.5,.5]])  #<- this is weight
        #network parameters:
            #self -- omitted 1
            #numOfLayers     2
            #numOfNeurons    3
            #inputSize       4
            #activation      5
            #loss            6
            #lr              7
            #weights         8
        #                          2, 3              4  5  6  7   8
        neural_net = NeuralNetwork(1, np.array([1]), 2, 1, 0, lr, w)
        ret = neural_net.calculate(x)
        print('ret:', ret)
        #neural_net.train(x, e_o)
        print('learn xor')
        