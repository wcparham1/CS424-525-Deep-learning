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

        #append 1 to represent bias
        #self.weights = np.append(self.weights, 1)
        
    #This method returns the activation of the net (net same as input for neuron)
    def activate(self,net):
        
        #first check for which activation function the neuron is using
        if (self.activation == 0):
            #save output for backpropagation
            self.output = net
            return net
        
        elif (self.activation == 1):
            #save output for backpropagation
            self.output = (1/(1+np.exp(-net)))
            #print("logistic activation")
            return 1/(1+np.exp(-net))
        
    #Calculate the output of the neuron should save the input and output for back-propagation.  --Calculate calls activate-- 
    def calculate(self,input):
        #each input will correspond with each weight in the same place in the weights array
        #find the sum of the inputs * weights
        net = 0        
        sum = 0
        for i in range(0, (len(input))):
            #net += round((input[i] * self.weights[i]), 4)
            sum = input[i] * self.weights[i]
            net += sum
            print('this is input[i]: ', input[i], ' this is self.weights[i]: ', self.weights[i], 'this is weight * input: ', input[i]*self.weights[i], ' this is net: ',net)
        
        #add bias weight
        #net += self.weights[-1] * input[-1]
        
        #save input for backpropagation
        self.n_input = net
        
        #print('net before rounding: ', net)
        #net = round(net, 2)
        #print('net after rounding: ', net)
       
        print('this is net before activation: ', net)
        output = self.activate(net)
        print('this is result of activate: ', output)
        return output
     
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        
        #self.n_input = net
        if(self.activation == 0):
            return (1)
        
        elif(self.activation == 1):
            return (np.exp(self.n_input))/((np.exp(self.n_input)+1)**2)
            
        print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        dE_dw = np.zeros(self.input_num+1)
        dE_dO = np.sum(wtimesdelta)
        dO_dnet = self.activationderivative()

        delta = dE_dO * dO_dnet
        for i in range(self.input_num):
           # dE_dw = delta * weights
        print('calcpartialderivative') 

        #return delta*weights
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        print('updateweight')
        
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
        
        for i in range (0, len(self.neuron_list)):
            print('\n')
            temp = self.neuron_list[i].calculate(input)
            neuron_calculations.append(temp)
    
        return neuron_calculations
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        ret = np.zeros((self.numOfNeurons, self.input_num))
        for i, neuron in enumerate(self.neuron_list):
            neuron_delta = neuron.calcpartialderivative(wtimesdelta[:,i])
            ret[i,:] = neuron_delta
            # neuron.updateweight()
        return ret
        print('calcwdeltas') 
    
    def print_info(self):
        print("fully_connected: num neurons: ", self.numOfNeurons, ' activation: ', self.activation, ' input_num: ', self.input_num, ' lr: ', self.lr, ' weights: ', self.weights)
           
           
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.layer_list = []
        
        if(weights is None):
            weights = np.random.randint(1,9,(self.numOfLayers*self.numOfNeurons*self.inputSize))
            self.weights = np.reshape(weights, [self.numOfLayers,self.numOfNeurons,self.inputSize])
        else:
            self.weights = weights

        for i in range(0, self.numOfLayers):
            if(self.numOfLayers <= 1):
                new_layer = FullyConnected(self.numOfNeurons, self.activation, self.inputSize, self.lr, self.weights)

            else:
                new_layer = FullyConnected(self.numOfNeurons, self.activation, self.inputSize, self.lr, self.weights[i]) 
            
            #print(new_layer.weights)
            self.layer_list.append(new_layer)
            
        #print('constructor') 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,x):
        print('this is input: ', x)
        x = np.array(x)
        current_input = x
        for i in range(0, len(self.layer_list)):
            #print('this is current input:', current_input)
            # current_input.append(1.0)
            current_input = np.concatenate((current_input, [1.0]))
            current_input = self.layer_list[i].calculate(current_input)
        
        self.y_hat = current_input
        return current_input
        print('calculate')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # MSE
        self.loss = np.mean(np.square(yp - y))
        print('calculateloss')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        return -(y-yp)
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        self.calculate(x)
        self.calculateloss(self.y_hat, y)
        delta = np.empty((1,self.numOfNeurons))

        for i in range(self.numOfNeurons):
            delta[:,i] = self.lossderiv(self.y_hat[i], y[i])
        # print("delta", delta)
        for j in reversed(range(len(self.layer_list))):
            delta = self.layer_list[i].calcwdeltas(delta)
            print("delta", j, delta)

        print('train')
    
    def print_info(self):
        print(self.numOfLayers,' ',self.numOfNeurons,' ',self.inputSize,' ',self.activation,' ',self.loss,' ',self.lr)

if __name__=="__main__":
    if (len(sys.argv)<2):
        #network parameters:
            #self -- omitted
            #numOfLayers 
            #numOfNeurons
            #inputSize
            #activation
            #loss
            #lr
            #weights
            
        #change inputs, we are using the incorrect numbers as input, and change number of neurons in the layer to be an array.
        #old weights: [[[.15,.25,.35],[.20,.30,.35]],[[.40,.50,.6],[.45,.55,.6]]]
        neural_net = NeuralNetwork(2, 2, 2, 1, 0, 0.5, [[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        # ret = neural_net.calculate([.05,.10])
        # print(ret)
        neural_net.train(np.array([0.05, .10]), np.array([.01, .99]))
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1]) #changed from x== to x=
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')
        
        
#logic that was in main
        #print('a good place to test different parts of your code')
        #fully_connected: def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        #neuron: def __init__(self,activation, input_num, lr, weights=None):
        #ner1 = Neuron(0, 3, 0, [1,5,2])
        #ner1.calculate([2,2,3])
        #flayer = FullyConnected(3, 0, 3, .5, [[1,5,2],[2,7,7],[3,9,1]])
        #flayer.calculate([2,2,3])
        #def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        
#fully connected layer math
        '''
        #initialize results matrix to hold all weight * input values
        w, h = len(input), (int)((len(self.weights)-1)/len(input))
        results = [[0 for x in range(w)] for y in range(h)]
        
        #create a matrix to hold all values aside from bias
        mat_no_bias = []        
        for i in range(0, (len(self.weights)-1)):
            mat_no_bias.append(self.weights[i])
        
        #reshape matrix for easier computation
        mat_no_bias = np.array(mat_no_bias)
        mat_no_bias = mat_no_bias.reshape(len(input),(int)((len(self.weights)-1)/len(input)))
        
        #find each value in results matrix
        for i in range(0, len(mat_no_bias)):
            for j in range(0, len(mat_no_bias[i])):
                
                results[i][j] = mat_no_bias[i][j] * input[i]
                print(mat_no_bias[i][j])
        
        #transpose results matrix to sum each row
        results = np.transpose(results)
        
        #find the sum of each row of the matrix
        for i in range(0, len(results)):
            row_sum = 0
            for j in range(0, len(results[i])):
                row_sum += results[i][j]
            
            #adding bias
            row_sum += 1
            neuron_calculations[i] = row_sum
        '''