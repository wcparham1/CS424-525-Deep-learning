#William C. Parham, Coby White
#Project 1 code provided by Lekpam Nkawula
#Lab 2 Convolution Neural Network
#March 1, 2023

import numpy as np
import sys
import matplotlib.pyplot as plt 
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, 
    #learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):   
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights 
        self.net = 0               
        self.input = 0             
        self.output = 0             # output for neutron (after activation)
        self.dEdw = 0               # partial deriv of weights for neuron
        
    #This method returns the activation of the net
    def activate(self,net):

        if self.activation == 0:
            return net

        return 1/(1+np.exp(-net)) 
        
    #Calculate the output of the neuron should save the input and 
    #output for back-propagation.   
    def calculate(self,input):

        '''
        self.input = np.array(input)
        self.net = np.matmul(self.input,self.weights)
        #print('net: ', self.net)
        self.net = np.sum(self.net)
        self.output = self.activate(self.net)
        #print('output: ', self.output)
        '''
        print('this is input in neuron:', input)
        input_by_weight_sum = 0
        
        for i in range(0, len(self.weights)):
            input_by_weight_sum += self.weights[i] * input[i]
        
        self.net = input_by_weight_sum
        print('this is self.net: ', self.net)
        self.output = self.activate(self.net)
        print('this is self.output: ', self.output)
        
        return self.output
    
    #special class that performs convolution on a neuron level
    def cnn_calculate(self, input):
        
        self.input = input
        res = []
        for i in range(0, len(input)):
            for j in range(0, len(input[i])):
                res.append(input[i][j] * self.weights[i][j])
        
        #Maybe account for bias here? <-- Unsure
        res.append(1)
        self.net = np.sum(res) 
        self.output = self.activate(self.net)
        
        return self.output

    #This method returns the derivative of the activation function 
    #with respect to the net   
    def activationderivative(self):

        if self.activation == 0:
            return 1

        activationderiv = np.exp(-self.net)/(1+np.exp(-self.net))**2  
        return activationderiv
    
    #This method calculates the partial derivative for each weight 
    #and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        delta = np.array(wtimesdelta*self.activationderivative())
        self.dEdw = delta*self.input
        wtimesdelta = delta*self.weights

        return wtimesdelta 
    
    #Simply update the weights using the partial derivatives and 
    # the learning weight
    def updateweight(self):
        self.weights = self.weights - self.lr*self.dEdw
    
    
    
#A class which represents a convolutional layer
class ConvolutionalLayer:
    #initialize with number of kernels in the layer, size of the kernel(assume square)
    #activation function for all neurons in layer, dimensions of input, learning rate
    #possibly a vector of weights(if not set random)
    def __init__(self, num_kernels, kernel_size, activation, input_dimensions, learning_rate, weights = None):
        self.num_kernels = num_kernels #number of kernels in layer
        self.kernel_size = kernel_size #size of each kernel
        self.activation = activation #activation function
        self.input_dimensions = input_dimensions #dimensions of input
        self.learning_rate = learning_rate #learning rate
        self.padding = 1
        self.stride = 1
        self.weights = [] #vector of weights for each neuron
        self.neurons = [] #vector to hold each neuron
        self.kernel_weights = [] #vector that holds groups of kernel weights
        self.kernels = []  # <-- unused fossil too nervous to delete this
        
        #if weights is uninitialized initialize randomly
        if(weights is None):
            num_weights = (((self.kernel_size * self.kernel_size)) * self.num_kernels)
            for i in range(0, num_kernels):
                x = []
                for j in range(0, int(num_weights/num_kernels)):
                    x.append(np.random.rand())
                    
                self.weights.append(x)
        else:
            #we are going to assume (hardcode) that the weights be passed in the same shape as our kernel.
            self.weights = weights
        
        #maybe the weight matrix is the kernel
        #and each neuron in the output is attached to a kernel that shares these weights?
        #now initialize each neuron in the layer
        
        #the formula for number of neurons in a layer is:
        num_neurons_in_layer = ((((self.input_dimensions[0] - kernel_size) / 1) + 1) * (((self.input_dimensions[1] - kernel_size) / 1) + 1)) * num_kernels
        self.output_height = ((self.input_dimensions[0] - kernel_size) / 1) + 1
        self.output_length = ((self.input_dimensions[1] - kernel_size) / 1) + 1
        
        self.num_neurons_in_layer = num_neurons_in_layer
        
        #set weights for each particular kernel, this will make neuron weight assignment easier
        if(num_kernels == 1):
            self.kernel_weights.append(self.weights)
        else:
            for i in range(num_kernels):
                    self.kernel_weights.append(self.weights[i])
        
        #calculate how many neurons per channel
        #print('neurons per layer: ', num_neurons_in_layer, 'num kernels: ', num_kernels)
        neurons_per_channel = int(num_neurons_in_layer / num_kernels) #################################### CASTING AS FLOAT HERE #########################################
        
        #append each neuron with the same weight
        for i in range(0, len(self.kernel_weights)):
            x = []
            for j in range(0, neurons_per_channel):
                reshaped_weights = np.reshape(self.kernel_weights[i], (self.kernel_size, self.kernel_size))
                neuron = Neuron(activation, 1, self.learning_rate, reshaped_weights)
                self.neurons.append(neuron)
                x.append(neuron)
                
            self.kernels.append(x)
            
        
    #calculate the activation of a cnn layer
    def calculate(self, input):  
        #the input to each neuron should be the values that correspond to the kernel entries
        #multiplied together then sent in to the neuron calculate.
        #print(self.input_dimensions)
        
        ret = []
        local_maps = []
        #print('dimensions of input: ', np.ndim(input), '----- in conv layer!')
        #calculate the output of the convolutional layer
        if(np.ndim(input) < 3):
            for i in range(0, len(self.kernels)):
                activation_res = []
                x_off = 0
                y_off = 0
                count = 0
                for neuron in self.kernels[i]:
                    in_vals = []
                    #in_vals = input[x_off:self.kernel_size+1, y_off:self.kernel_size+1]
                    for x in range(0, self.kernel_size):
                        for y in range(0, self.kernel_size):
                            in_vals = np.append(in_vals, (input[x + x_off][y + y_off]))

                    #offset rules --- probably works    
                    if y_off < self.output_length:
                        y_off += 1
                    if y_off == self.output_length:
                        y_off = 0
                        x_off += 1

                    #reshape arrays for visualization
                    in_vals = np.array(in_vals)
                    in_vals = np.reshape(in_vals, (self.kernel_size, self.kernel_size))
                    activate_return = neuron.cnn_calculate(in_vals) 
                    activation_res.append(activate_return)

                    count += 1
                    
                #reshape output to help with visualization and input
                activation_res = np.reshape(activation_res, (int(self.output_height), int(self.output_length)))
                #print('activation res:\n', activation_res)
                ret.append(activation_res)
            return ret

        
        #terrible way of handling how to calculate input when working with more than 1 input feature map.
        else:
            for z in range(0, len(input)):
                for i in range(0, len(self.kernels)):
                    activation_res = []
                    x_off = 0
                    y_off = 0
                    count = 0
                    for neuron in self.kernels[i]:
                        in_vals = []
                        #in_vals = input[x_off:self.kernel_size+1, y_off:self.kernel_size+1]
                        for x in range(0, self.kernel_size):
                            for y in range(0, self.kernel_size):
                                in_vals = np.append(in_vals, (input[z][x + x_off][y + y_off]))

                        #offset rules --- probably works    
                        if y_off < self.output_length:
                            y_off += 1
                        if y_off == self.output_length:
                            y_off = 0
                            x_off += 1

                        #reshape arrays for visualization
                        in_vals = np.array(in_vals)
                        in_vals = np.reshape(in_vals, (self.kernel_size, self.kernel_size))
                        
                        #find the activation of the neuron
                        activate_return = neuron.cnn_calculate(in_vals) 
                        activation_res.append(activate_return)

                        count += 1
                        
                    #reshape output to help with visualization and input
                    #activation_res = np.reshape(activation_res, (int(self.output_height), int(self.output_length)))
                    local_maps.append(activation_res)
                    
            temp1 = local_maps[0]
            temp2 = local_maps[1]
            
            ret = []
            for i in range(0,len(temp1)):
                sum = temp1[i] + temp2[i]
                #print('sum: ', sum, i)
                ret.append(sum)

            ret = np.array(ret)
            ret = np.reshape(ret, (int(self.output_height), int(self.output_length)))
            
            return ret
        #return ret
            
            

    #Helper function to print info for the layer        
    def print_info(self):
        print('The following data is inside the convolutional layer: \n') 
        print('num_kernels: ', self.num_kernels)
        print('kernel_size: ', self.kernel_size)
        print('activation: ', self.activation)
        print('input_dimensions: ', self.input_dimensions) 
        print('num_kernels: ', self.learning_rate) 
        print('padding: ', self.padding)
        print('stride: ', self.stride)
        print('weights: ', self.weights)
        print('num neurons in layer: ', self.num_neurons_in_layer)
        print('kernel_specific_weights: ', self.kernel_weights)
        print('neuron weights: \n')
        
        count = 0
        for x in self.neurons:
            print(count, ' ', x.weights,'\n')
            count += 1
     
     
            
#A class which represents a max pooling layer
class MaxPoolingLayer:
    
    #Max pooling initialization
    def __init__(self, kernel_size, input_dimensions):
        self.kernel_size = kernel_size
        self.input_dims = input_dimensions
        self.stride = kernel_size
        
    def calculate(self, input):
                    
        #determine how many times to loop
        self.output_height = ((self.input_dims[0] - self.kernel_size) / self.kernel_size) + 1
        self.output_length = ((self.input_dims[1] - self.kernel_size) / self.kernel_size) + 1
        counts = int(self.output_height * self.output_length)
     
        x_off = 0
        y_off = 0
        
        res_locs = []
        ret = []

        #find our maxes        
        for z in range (0, len(input)):
            frame_res = []
            for i in range(0, counts):
                in_vals = []
                x_off = 0
                y_off = 0
                
                for x in range(0, self.kernel_size):
                    for y in range(0, self.kernel_size):
                        in_vals = np.append(in_vals, (input[z][x + x_off][y + y_off]))
            
                if(y_off >= self.kernel_size):
                    y_off = 0
                    x_off += self.kernel_size
                elif(y_off < self.kernel_size):
                    y_off += self.kernel_size
                
                #append our max value
                #frame_res.append(np.max(in_vals))
                
                frame_res = np.append(frame_res, np.max(in_vals))

            #reshape frame results and append to return value
            frame_res = np.reshape(frame_res,(int(self.output_length), int(self.output_height))) 
            #print(frame_res)           
            ret.append(frame_res)
        
        #loop through all our entries in ret and find the location of the max.
        max_locs = []
        for index,channel in enumerate(ret):
            local_max = []
            for row in channel:
                for entry in row:
                    rows,cols = np.where(input[index] == entry)
                    rc_tuppy = [rows, cols]
                    local_max.append(rc_tuppy)
            max_locs.append(local_max)
                    
        #save location of maxes        
        self.max_locations = max_locs

        #return our pooling layer output!
        return ret



#A class which represents a flatten layer
class FlattenLayer: 
    
    def __init__(self, input_size):
        self.input_size = input_size
    
    def calculate(self, input):
        
        length = int(self.input_size[0] * self.input_size[1] * self.input_size[2])
        res = np.resize(input, (length, 1))
        
        return res
          
        
          
#A fully connected layer        
class FullyConnected:
   
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights 

        #print('this is weights: ', self.weights)
        self.Neurons = []
        for i in range(self.numOfNeurons):
            self.Neurons.append(Neuron(self.activation,self.input_num,self.lr,self.weights)) #changed from appending weights[i] #3/6/2023 because we only add one layer at a time.
        
    #calculate the output of all the neurons in the layer and 
    # return a vector with those values (go through the neurons 
    # and call the calculate() method)      
    def calculate(self, input):
        #print('this is input: ', input)
        results = []
        for neuron in self.Neurons:
            results.append(neuron.calculate(input))
        
        results.append(1)
        return results 
            
    # given the next layer's w*delta, should run through the neurons 
    # calling calcpartialderivative() for each (with the correct value), 
    # sum up its ownw*delta, and then update the weights s
    # (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        wtimesdeltas = []

        for i in range(self.numOfNeurons):
            wtimesdeltas.append(self.Neurons[i].calcpartialderivative(wtimesdelta[i]))
            self.Neurons[i].updateweight()

        return np.sum(wtimesdeltas,axis=0)


        
#An entire neural network        
class NeuralNetwork:
    
    #initialize with the number of layers, number of neurons in each layer (vector), 
    # input size, activation (for each layer), the loss function, the learning rate 
    # and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, inputSize, loss, lr):
        
        #1.a) Initialize the network with the input size, loss function, and learning rate.
        self.inputSize = inputSize 
        self.loss = loss
        self.lr = lr
        self.Layers = []

        
    #add layer will add layers
    #1.b) Add a method to the class (NeuralNetwork) called addLayer.  This method will accept all details to initialize the layer depending on type of layer.
    #numOfNeurons, input_num, correspond only to fc layer
    #num_kernels, kernel_size, input_dimensions correspond only to cn layer
    #we can toggle between layer type by specifying cnn_layer
    def addLayer(self, layer_type, activation=None, lr=None, weights=None, numOfNeurons=None, input_num=None, num_kernels=None, kernel_size=None, input_dimensions=None):
        
        if(layer_type == 'fc'):
            #we will append the new fully connected layer with the inputted values.  
            #This will reflect how keras adds layers.
            self.Layers.append(FullyConnected(numOfNeurons, activation, input_num, lr, weights))
        elif(layer_type == 'cn'):
            #we will append the new convolutional layer with the inputted values.
            #this will reflect how keras adds layers.
            self.Layers.append(ConvolutionalLayer(num_kernels, kernel_size, activation, input_dimensions, lr, weights))
        elif(layer_type == 'flat'):
            #we will append a flatten layer to our network.
            self.Layers.append(FlattenLayer(input_dimensions))
        elif(layer_type == 'max_pool'):
            #we will add a max pooling layer
            self.Layers.append(MaxPoolingLayer(kernel_size=kernel_size, input_dimensions=input_dimensions))
        else:
            print('please specify what type of layer you would like.\n cnn_layer = False for fc, True for Cn')
            
        
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        results = self.Layers[0].calculate(input)

        for i in range(1,len(self.Layers)):
            results = self.Layers[i].calculate(results)

        return results[:-1]

    #Given a predicted output and ground truth output simply 
    # return the loss (depending on the loss function)
    def calculateloss(self,y,yp):
        if self.loss == 0:
            return 0.5*np.sum((y-yp)**2)
        
        res = []
        for k in range(len(y)):
            res.append([-(i*np.log2(j)+(1-i)*np.log2(1-j)) for i,j in zip(y[k],yp[k])])

        return np.mean(res)
    
    #Given a predicted output and ground truth output simply 
    # return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,y,yp):
        if self.loss == 0:
            return -(y-yp)

        return [-i/j + (1-i)/(1-j) for i,j in zip(y,yp)]
    
    #Given a single input and desired output preform one step of backpropagation 
    #(including a forward pass, getting the derivative of the loss, 
    #and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        # feedforward 
        yp = self.calculate(x)

        # backpropagate 
        wtimesdelta = self.Layers[-1].calcwdeltas(self.lossderiv(y,yp))

        for i in range(self.numOfLayers-2,-1,-1):
            wtimesdelta = self.Layers[i].calcwdeltas(wtimesdelta)



#Main
if __name__=="__main__":
    if (len(sys.argv)<2):
        print('usage: python project2.py [example1|example2|example3]')
        
    
    #def addLayer(self, activation, lr, weights, cnn_layer, numOfNeurons=None,  input_num=None,  num_kernels=None, kernel_size=None, input_dimensions=None):
    elif (sys.argv[1] == 'example1'):
        print('Running example 1 from the lab write up.')
        
        w = np.array([[1,2,3],[4,5,6],[7,8,9]])
        num_kers = 1
        ker_size = 3
        a_func = 1
        input_dims = np.array([4,4,1])
        lr = 0.3
        input = np.array([[1,2,3,4,5],[5,6,7,8,8],[9,10,11,12,13],[13,14,15,16,17],[17,18,19,20,21]])

        neural_net = NeuralNetwork(np.array([5,5,1]), 1, 0.3)
        neural_net.addLayer(layer_type='cn', activation=a_func, lr=lr, weights=w, num_kernels=1, kernel_size=3, input_dimensions=np.array([5,5,1]))
        neural_net.addLayer(layer_type='flat', input_dimensions=np.array([3,3,1]))
        neural_net.addLayer(layer_type='fc', numOfNeurons=1, activation=a_func, lr=lr, input_num=9, weights=np.array([1,2,3,4,5,6,7,8,9]))
        
        print(neural_net.calculate(input))
    
    
    elif(sys.argv[1] == 'example2'):
        print('Running example 2 from the lab write up.')
        
        w = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]])
        a_func = 1
        input_dims = np.array([7,7,1])
        lr = 0.3
        input = np.array([[1,2, 3, 4, 5, 6, 7],
                          [ 8, 9,10,11,12,13,14],
                          [15,16,17,18,19,20,21],
                          [22,23,24,25,26,27,28],
                          [29,30,31,32,33,34,35],
                          [36,37,38,39,40,41,42],
                          [43,44,45,46,47,48,49]])

        neural_net = NeuralNetwork(np.array([7,7,1]), 1, 0.3)
        neural_net.addLayer(layer_type='cn', activation=a_func, lr=lr, weights=w, num_kernels=2, kernel_size=3, input_dimensions=np.array([7,7,2]))
        w = np.array([[0,0,0],[0,1,0],[0,0,0]])
        neural_net.addLayer(layer_type='cn', activation=a_func, lr=lr, weights=w, num_kernels=1, kernel_size=3, input_dimensions=np.array([5,5,2]))
        neural_net.addLayer(layer_type='flat', input_dimensions=np.array([3,3,1]))
        neural_net.addLayer(layer_type='fc', numOfNeurons=1, activation=a_func, lr=lr, input_num=9, weights=np.array([1,2,3,4,5,6,7,8,9]))

        print(neural_net.calculate(input))
        
    elif(sys.argv[1] == 'example3'):
        print('Running example 3 from the lab write up.')
        
        w = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
        a_func = 1
        input_dims = np.array([7,7,1])
        lr = 0.3

        input = np.array([[1, 2, 3, 4, 5 ,6, 7, 8],
                          [ 9,10,11,12,13,14,15,16],
                          [17,18,19,20,21,22,23,24],
                          [25,26,27,28,29,30,31,32],
                          [33,34,35,36,37,38,39,40],
                          [41,42,43,44,45,46,47,48],
                          [49,50,51,52,53,54,55,56],
                          [57,58,59,60,61,62,63,64]])
        

        neural_net = NeuralNetwork(np.array([8,8,1]), 1, 0.3)
        neural_net.addLayer(layer_type='cn', activation=a_func, lr=lr, weights=w, num_kernels=2, kernel_size=3, input_dimensions=np.array([7,7,2]))
        neural_net.addLayer(layer_type='max_pool', kernel_size=2, input_dimensions=np.array([6,6,2]))
        neural_net.addLayer(layer_type='flat', input_dimensions=np.array([3,3,2]))
        neural_net.addLayer(layer_type='fc', numOfNeurons=1, activation=a_func, lr=lr, input_num=18, weights=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]))

        print(neural_net.calculate(input))
        