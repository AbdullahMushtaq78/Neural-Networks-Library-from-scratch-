# Krypton, A Deep Learning Library
## Introduction
Welcome to the Krypton, A Deep Learning Library repository! This repository houses a powerful Python library for constructing and training neural networks, with a focus on deep learning tasks. The library provides a comprehensive set of features and functionalities to facilitate the development of robust machine learning models from scratch.

## Features
- **Dense Layers**:
    - Easily create fully connected layers with customizable input and output sizes.
- **Activation Functions**: 
    - Variety of activation functions, including ReLU, Softmax, Sigmoid, and Tanh, to introduce non-linearity and enhance model performance.
- **Loss Functions**: 
    - Select from Binary Cross-Entropy (BCE) and Categorical Cross-Entropy (CCE) loss functions to efficiently optimize model parameters.
- **Optimizers**:
    - Utilize popular optimization algorithms like Stochastic Gradient Descent (SGD) with Momentum, AdaGrad, RMSprop, and Adam to achieve faster convergence and improved training results.
- **Backpropagation**:
    - Benefit from the implementation of complete backpropagation, enabling automatic computation of gradients for efficient model parameter updates.

## Installation
To use the Deep Learning Library, follow these steps:

- Ensure that you have Numpy installed. If not, run **pip install numpy**.
- Clone this repository to your local machine using git clone [[Krypton Repository URL](https://github.com/AbdullahMushtaq78/Neural-Networks-Library-from-scratch-)].
- Import the library in your Python script: **import krypton as kp**.
- Start building and training your neural networks with ease!
## Getting Started
To get started with the Deep Learning Library, refer to the provided examples in the int main section of this library. The examples demonstrate how to create and train neural networks using the library's various features. The comments along code provides detailed explanations of each feature.

## Contributions
Contributions to the Deep Learning Library are highly appreciated! If you encounter any issues or have suggestions for improvements, please submit an issue on the repository's issue tracker. Feel free to fork the repository and submit pull requests to enhance the library's functionality.
## Experiments
In order to validate the working and performance of the library, MNIST dataset is used to calculate different metrics:

###### Accuracies:
- Training Accuracy = 99.97%, Training Loss: 0.080
- Validation Accuracy = 98%, Validation Loss: 0.13
- Testing Accuracy = 97.9%, Testing Loss: 0.138
###### Training, Validation, and Testing Curves:
![image](https://github.com/AbdullahMushtaq78/Neural-Networks-Library-from-scratch-/assets/96788451/031df827-7463-42fc-b26c-5cacbfea1709)
###### Confusion Matrix:
![image](https://github.com/AbdullahMushtaq78/Neural-Networks-Library-from-scratch-/assets/96788451/637c401d-5cb1-4a04-91da-74d770d3c94a)
###### Experimental Configurations:
- Layer 1 size 784
- Layer 2 size 256
- Layer 3 size 128
- Layer 4 size 10
- Learning rate = 0.1
- Decay rate = 1e-7
- Momentum = 0.9
- Activations =
    - Sigmoid
    - Sigmoid
    - Softmax
- Loss = Categorical Cross Entropy
- Optimizer = SGD with momentum
- Epochs = 200
- Mean Subtraction = True
- Batch size = 64

