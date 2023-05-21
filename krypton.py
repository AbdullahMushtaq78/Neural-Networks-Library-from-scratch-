
#*####################################################################################\n
#*                                      Welcome                                      #\n
#*                 Neural Networks Deep Learning Library from Scratch                #\n
#!                                  Project Krypton                                  #\n
#*####################################################################################\n
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()
class Layer_Dense:
    def __init__(self,n_inputs, n_neurons,layer_name=None):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        if layer_name is not None:
            self.layer_name = layer_name
        else:
            self.layer_name = "Layer"
        '''
            -> multiplying by 0.10 to restrict gaussian distribution between -1 to +1
            -> The shape of the weights is designed in such a way that we do not need to take Transpose when doing the forward pass.
        '''
    def __str__(self):
        
        description = self.layer_name + " : weights" + str(self.weights.shape) + ", biases: " + str(self.biases.shape)
                
        return  description
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights)+ self.biases
    def backward(self, derivatives_next_layer): 
        '''
        #*dvalues are passed in gradients from the next layer
        #*Gradients on parameters
        #* Here we are calculating the gradients/derivatives on both weights and biases to be used in optimization procedure.
        '''
        self.dweights = np.dot(self.inputs.T, derivatives_next_layer) 
        self.dbiases = np.sum(derivatives_next_layer, axis=0, keepdims=True)
        #Gradients on values
        '''
        #* Here we are calculating the gradients we will be using as  dvalues for previous layers.
        #* This will help us to just use this values for optimization of previous layers. Because we dont have to calculate it again by calling the values of this layer from another layer.
        '''
        self.dinputs = np.dot(derivatives_next_layer, self.weights.T) 
        #* This equation corresponds to the backprop optimization. Because we are eliminating the unnecessary compuation and using only the required values.
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs #*Saving inputs so that we can them later for optimization
        self.output = np.maximum(0, inputs) #* Main softmax function.
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() 
        self.dinputs[self.inputs <= 0] = 0 
        #*Zero gradient values where input values are negative, Simple Step function for derivatives.
        #*These dinputs can be used to compute derivatives and optimize the parameters of prevoiusly connected layers.  
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs #*Saving inputs so that we can them later for optimization
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #* Calculating the exponential values of input
        #* Here we are subtracting the maximum value because we want to minimize the chances of overflow happening at some point when the input to exponential function is very large
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        #* calcualting the probabilities by dividing with the summation of all exponential values. This will scale the probabilities between 0 and 1.
        self.output = probabilities #* Saving the probabilities for later use in optimization
    def backward(self, derivative_next_layer): 
        #* Here the derivative_next_layer is the derivatives coming from the Loss function. Because we mostly use Softmax at the end of the network
        self.dinputs = np.empty_like(derivative_next_layer) #* Here we are making an empty array with same shape as derivative_next_layer because there is not change in the no of neurons or something at this point.
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, derivative_next_layer)):
        # Flatten output array
            single_output = single_output.reshape(-1, 1)
            #* Here we are calculating the jacobian matrix in order to calculate the gradients of the output, We doing it in iterative way we want a separate Jacobian after each iteration.
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # *Calculating the sample-wise gradient and adding it to the array of sample gradients which will be used for optimization 
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
class Activation_Sigmoid:
    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(inputs)
    def backward(self, dvalues):
        #* Sigmoid (1 - Sigmoid) but because we need to do back propagation,so we need to multiply it with dvalues
        self.dinputs = dvalues * (1 - self.output) * self.output
class Activation_Tanh:
    def Tanh(self, inputs):
        inputs = np.array(inputs, dtype=np.float16)
        return (np.exp(inputs)-np.exp(-inputs))/(np.exp(inputs)+np.exp(-inputs))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.Tanh(inputs)
    def backward(self, dvalues):
        #* Sigmoid (1 - Sigmoid) but because we need to do back propagation,so we need to multiply it with dvalues
        self.dinputs = dvalues * (1 - np.square(self.output))
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        sample_losses  = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_values = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_values - (1 - y_true) / (1 - clipped_values)) / outputs
        self.dinputs = self.dinputs / samples 
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred) #* Saving the number of samples passed as arguments
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #* Clipping the predictions between 0.00000001 and 0.99999999
        if len(y_true.shape) == 1: #*Scalar Values 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: #*One hot encoded ground truths
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            '''
            #*Making all the non-relevant values equal to zero and 
            #*only non-zero values which remains in this array/matrix would be the classes that we are interested in
            '''
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues) #* Saving the number of samples
        labels = len(dvalues[0]) #*Calculating the length of labels
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    '''
        Making a combined activation and Loss function class
    '''
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1
        self.dinputs = self.dinputs / samples
class Optimizer_SGD:
    def __init__(self,learning_rate = 1.0, decay = 0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. +self.decay* self.iterations))
            '''
                Here are updating the learning rate for each iteration, it is using the Deca Rate OR Exponential Decaying formula with some slight modifications
                Updating the learning rate by taking the reciprocal of step count fraction. This fraction can be changed as per passed as a parameter, hence it is a hyperparameter "Learning Rate Decay".
                It takes the step/iteration and multiplies it with decay rate. As the alogirthms works it can produce a large step in the learning rate, hence we will take the reciprocal of this value
                And multiply it with the initial learning rate value. Adding 1 in denominator because we dont want to increase the learning rate very large. 1/0.001 = 1000 but adding 1 in 0.001 makes sure that it never produces a large value. (1/1.001 = 0.999)
                '''
    def update_params(self,layer):
        
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
        
        
        # '''
        # Update the parameters by taking each layer
        # Here it is taking each layer and updating its parameters by taking the differentiated parameters and multiplying them by learning rate 
        # '''
        # layer.weights += -self.current_learning_rate * layer.dweights
        # layer.biases += -self.current_learning_rate * layer.dbiases
    def post_update_params(self):
        '''
        Just updating the iteration after each update, because we want the learning rate decay feature to work properly.
        '''
        self.iterations += 1

class Optimizer_AdaGrad:
    def __init__(self,learning_rate = 1.0, decay = 0., epsilon= 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. +self.decay* self.iterations))
            '''
                Here are updating the learning rate for each iteration, it is using the Deca Rate OR Exponential Decaying formula with some slight modifications
                Updating the learning rate by taking the reciprocal of step count fraction. This fraction can be changed as per passed as a parameter, hence it is a hyperparameter "Learning Rate Decay".
                It takes the step/iteration and multiplies it with decay rate. As the alogirthms works it can produce a large step in the learning rate, hence we will take the reciprocal of this value
                And multiply it with the initial learning rate value. Adding 1 in denominator because we dont want to increase the learning rate very large. 1/0.001 = 1000 but adding 1 in 0.001 makes sure that it never produces a large value. (1/1.001 = 0.999)
                '''
    def update_params(self,layer):
        
        if not hasattr(layer, "weight_cache_history"):
            layer.weight_cache_history= np.zeros_like(layer.weights)
            layer.biases_cache_history = np.zeros_like(layer.biases)
        layer.weight_cache_history += np.square(layer.dweights)
        layer.biases_cache_history += np.square(layer.dbiases)

        
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache_history) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache_history) + self.epsilon)
    
    
        # '''
        # Update the parameters by taking each layer
        # Here it is taking each layer and updating its parameters by taking the differentiated parameters and multiplying them by learning rate 
        # '''
        # layer.weights += -self.current_learning_rate * layer.dweights
        # layer.biases += -self.current_learning_rate * layer.dbiases
    def post_update_params(self):
        '''
        Just updating the iteration after each update, because we want the learning rate decay feature to work properly.
        '''
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self,learning_rate = 0.001, decay = 0., epsilon= 1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. +self.decay* self.iterations))
            '''
                Here are updating the learning rate for each iteration, it is using the Deca Rate OR Exponential Decaying formula with some slight modifications
                Updating the learning rate by taking the reciprocal of step count fraction. This fraction can be changed as per passed as a parameter, hence it is a hyperparameter "Learning Rate Decay".
                It takes the step/iteration and multiplies it with decay rate. As the alogirthms works it can produce a large step in the learning rate, hence we will take the reciprocal of this value
                And multiply it with the initial learning rate value. Adding 1 in denominator because we dont want to increase the learning rate very large. 1/0.001 = 1000 but adding 1 in 0.001 makes sure that it never produces a large value. (1/1.001 = 0.999)
                '''
    def update_params(self,layer):
        

        
        if not hasattr(layer, "weight_cache_history"):
            layer.weight_cache_history,layer.biases_cache_history   = np.zeros_like(layer.weights), np.zeros_like(layer.biases)
            
        layer.weight_cache_history = self.rho * layer.weight_cache_history + ((1 - self.rho) * np.square(layer.dweights))
        layer.biases_cache_history = self.rho * layer.biases_cache_history + ((1 - self.rho) * np.square(layer.dbiases))
        
        layer.weights += -self.current_learning_rate * (layer.dweights / (np.sqrt(layer.weight_cache_history) + self.epsilon))
        layer.biases += -self.current_learning_rate * (layer.dbiases / (np.sqrt(layer.biases_cache_history) + self.epsilon))
    
    
        # '''
        # Update the parameters by taking each layer
        # Here it is taking each layer and updating its parameters by taking the differentiated parameters and multiplying them by learning rate 
        # '''
        # layer.weights += -self.current_learning_rate * layer.dweights
        # layer.biases += -self.current_learning_rate * layer.dbiases
    def post_update_params(self):
        '''
        Just updating the iteration after each update, because we want the learning rate decay feature to work properly.
        '''
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self,learning_rate = 0.001, decay = 0., epsilon= 1e-7, beta_1=0.9,beta_2=0.999 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. +self.decay* self.iterations))
            '''
                Here are updating the learning rate for each iteration, it is using the Deca Rate OR Exponential Decaying formula with some slight modifications
                Updating the learning rate by taking the reciprocal of step count fraction. This fraction can be changed as per passed as a parameter, hence it is a hyperparameter "Learning Rate Decay".
                It takes the step/iteration and multiplies it with decay rate. As the alogirthms works it can produce a large step in the learning rate, hence we will take the reciprocal of this value
                And multiply it with the initial learning rate value. Adding 1 in denominator because we dont want to increase the learning rate very large. 1/0.001 = 1000 but adding 1 in 0.001 makes sure that it never produces a large value. (1/1.001 = 0.999)
                '''
    def update_params(self,layer):
        
        if not hasattr(layer, "weights_cache_history"):
            layer.weights_cache_history,layer.biases_cache_history = np.zeros_like(layer.weights), np.zeros_like(layer.biases)
            layer.weights_momentums, layer.biases_momentums = np.zeros_like(layer.weights), np.zeros_like(layer.biases)
            
        layer.weights_momentums = self.beta_1 * layer.weights_momentums + (1 - self.beta_1) * layer.dweights
        layer.biases_momentums = self.beta_1 * layer.biases_momentums + (1 - self.beta_1) * layer.dbiases
        
        layer.weights_cache_history = self.beta_2 * layer.weights_cache_history + (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache_history = self.beta_2 * layer.biases_cache_history + (1 - self.beta_2) * layer.dbiases**2
        
        weights_momentums_optimized = layer.weights_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        biases_momentums_optimized = layer.biases_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        weights_cache_optimized = layer.weights_cache_history / (1 - self.beta_2 ** (self.iterations + 1))
        biases_cache_optimized = layer.biases_cache_history / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weights_momentums_optimized / (np.sqrt(weights_cache_optimized) + self.epsilon)
        layer.biases += -self.current_learning_rate * biases_momentums_optimized / (np.sqrt(biases_cache_optimized) + self.epsilon)
    

        # '''
        # Update the parameters by taking each layer
        # Here it is taking each layer and updating its parameters by taking the differentiated parameters and multiplying them by learning rate 
        # '''
        # layer.weights += -self.current_learning_rate * layer.dweights
        # layer.biases += -self.current_learning_rate * layer.dbiases
    def post_update_params(self):
        '''
        Just updating the iteration after each update, because we want the learning rate decay feature to work properly.
        '''
        self.iterations += 1


def train():
    X,y = spiral_data(samples=300, classes=2) #Creating Dataset
    y = y.reshape(-1,1)
    dense1 = Layer_Dense(2,64, layer_name="layer1") #Creating First Dense Layer for inputs 
    activation1 = Activation_ReLU() #Activation function for First Dense Layer
    dense2 = Layer_Dense(64,1, layer_name="layer2") #Creating Second Dense Layer for outputs
    activation2 = Activation_Sigmoid() #Creating Activation function and Loss function combination for output layer
    loss_fn = Loss_BinaryCrossEntropy()
    #SGD learning_rate 0.02, decay = 1e-5, momentum-0.999
    #RMSprop learning_rate=0.02,decay = 1e-4, rho=0.999
    #Adam learning_rate=0.05,decay = 1e-5
    optimizer = Optimizer_Adam(learning_rate=0.05,decay = 1e-5) #Craeting Optimizer for paramters optimization 
    accuracies = [] #Keeping record of accuracies after each epoch
    losses = [] #keeping record of losses after each epoch
    learning_rates =[] #keeping record of learning rates after each epoch
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        loss = loss_fn.calculate(activation2.output, y)

        #predictions = np.argmax(loss_fn.output, axis = 1)
        #if len(y.shape) == 2: #* If ground truth labels are one hot encoded, then we need to take the argmax row-wise.
        #    y = np.argmax(y, axis =1) #*This will give us the argmax row-wise for each instance and save them in y
        #accuracy = np.mean(predictions == y)
        #accuracies.append(accuracy)
        losses.append(loss)
        learning_rates.append(optimizer.current_learning_rate)
        #if not epoch %100:
            #print('epoch: ' + str(epoch) + ', acc: '+ str(accuracy) + ', loss: '+ str(loss), ', lr: ' + str(optimizer.current_learning_rate))
        #Backward Pass
        loss_fn.backward(activation2.output, y)
        #print(loss_fn.dinputs)
        activation2.backward(loss_fn.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.pre_update_params() #*Learning Rate Decay update
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
    plt.plot(accuracies,color= 'green',linewidth=3, label="Accuracy")
    plt.plot(losses,color= 'blue',linewidth=3, label="Loss")
    plt.plot(learning_rates,color= 'orange',linewidth=3, label="Learning Rate")
    plt.legend(loc="upper right",frameon=False)
    plt.show()



    # X_test, y_test = spiral_data(samples=100, classes=3)
    # # Perform a forward pass of our testing data through this layer
    # dense1.forward(X_test)
    # # Perform a forward pass through activation function
    # # takes the output of first dense layer here
    # activation1.forward(dense1.output)
    # # Perform a forward pass through second Dense layer
    # # takes outputs of activation function of first layer as inputs
    # dense2.forward(activation1.output)
    # # Perform a forward pass through the activation/loss function
    # # takes the output of second dense layer here and returns loss
    # loss = loss_fn.forward(dense2.output, y_test)
    # # Calculate accuracy from output of activation2 and targets
    # # calculate values along first axis
    # predictions = np.argmax(loss_fn.output, axis=1)
    # if len(y_test.shape) == 2:
    #     y_test = np.argmax(y_test, axis=1)
    # accuracy = np.mean(predictions==y_test)
    # print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == "__main__":
    train()
'''
#Basic forward model:
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

#layer 2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
'''
