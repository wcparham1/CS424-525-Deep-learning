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
            self.weights = [1] * self.input_num
            
            #initialize random weight values
            for i in range(0, len(self.weights)):
                self.weights[i] = np.random.randint(1,9)
        else:
            self.weights = weights
  
        #append 1 to represent bias
        self.weights = np.append(self.weights, 1)
        #print('constructor')    
        
    #This method returns the activation of the net (net same as input for neuron)
    def activate(self,net):
        
        #first check for which activation function the neuron is using
        if (self.activation == 0):
            #save output for backpropagation
            self.output = net
            return self.output
        
        elif (self.activation == 1):
            #save output for backpropagation
            self.output = (1/(1+np.exp(-net)))
            print("logistic activation")
            return self.n_output
        #print('activate')  
        
    #Calculate the output of the neuron should save the input and output for back-propagation.  --Calculate calls activate-- 
    def calculate(self,input):
        #each input will correspond with each weight in the same place in the weights array
        #find the sum of the inputs * weights
        net = 0        
        for i in range(0, len(input)):
            net += input[i] * self.weights[i]
        
        #add bias weight
        net += self.weights[len(self.weights) - 1]
        #save input for backpropagation
        self.n_input = net
        
        #print("this is net after looping: ", net)
        output = self.activate(net)

        #print('neuron_output: ', output)
        return output
     
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        print('calcpartialderivative') 
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        print('updateweight')
        
    def print_info(self):
        print('Neuron info: Activation: ', self.activation, ' input_num: ', self.input_num, ' learning rate: ', self.lr, ' weights: ', self.weights)
        

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    #3, 0, 3, .5, [[1,2,3],[5,7,9],[2,7,1]]
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
            neuron_calculations.append(self.neuron_list[i].calculate(input))
            
        
        print('fully_connected calculate: ', neuron_calculations) 
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
    
    def print_info(self):
        print("fully_connected: num neurons: ", self.numOfNeurons, ' activation: ', self.activation, ' input_num: ', self.input_num, ' lr: ', self.lr, ' weights: ', self.weights)
           
        
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
        #print('a good place to test different parts of your code')
        #fully_connected: def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        #neuron: def __init__(self,activation, input_num, lr, weights=None):
        #ner1 = Neuron(0, 3, 0, [1,5,2])
        #ner1.calculate([2,2,3])
        flayer = FullyConnected(3, 0, 3, .5, [[1,5,2],[2,7,7],[3,9,1]])
        flayer.calculate([2,2,3])
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1]) #changed from x== to x=
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')
        
        
        
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