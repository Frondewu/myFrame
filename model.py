# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
def gradient(x, dtype):

    x_grad = np.ones(x.shape)
    if dtype == 'sigmoid':
        x_grad = (x_grad-x)*x
    if dtype == 'relu':
        x_grad = np.where(x>0, x_grad, 0)        
    return x_grad

def activate(x, dtype):

    x_act = x
    if dtype == 'sigmoid':
        x_act = 1.0 / (1.0 + np.exp(-x))
    if dtype == 'relu':
        x_act = np.where(x>0, x_act, 0)
    return x_act

def softmax(x):

    exp_minmax = lambda x: np.exp(x-np.max(x))
    normalize = lambda x: x / np.sum(x)
    x = np.apply_along_axis(exp_minmax, 1, x)
    x = np.apply_along_axis(normalize, 1, x)
    return x

def im2col(data_im, batch_size, height, width, height_col, width_col, channels, ksize, stride, pad, pad_value=0):

    data_pad = np.pad(data_im,((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=pad_value)
    data_col = np.zeros((batch_size, height_col*width_col, ksize*ksize*channels))
    for h in range(height_col):
        for w in range(width_col):
            data_block = data_pad[:, h*stride:h*stride+ksize, w*stride:w*stride+ksize, :]
            data_col[:, h*width_col+w, :] = data_block.reshape(batch_size, -1)
    return data_col

def col2im(data_col, batch_size, height, width, height_col, width_col, channels, ksize, stride, pad):

    height_pad = height + 2 * pad
    width_pad = width + 2 * pad
    data_pad = np.zeros((batch_size, height_pad, width_pad, channels))
    for h in range(height_col):
        for w in range(width_col):
            data_pad[:, h*stride:h*stride+ksize, w*stride:w*stride+ksize, :] += data_col[:,h*width_col+w,:].reshape(batch_size, ksize, ksize, channels)
    data_im = data_pad[:,pad:height_pad-pad, pad:width_pad-pad, :]
    return data_im

def random_initializer(shape, min_x, max_x):
    x = np.random.uniform(min_x, max_x, shape)
    return x
def constant_initializer(shape, value):
    x = np.full(shape, value)
    return x

class ConnectedLayer:

    def __init__(self, weights, biases=None, activate_type=None, learning_rate=1e-3, momentum=0.9, is_normlize=False):
        self.weights = weights
        self.biases = biases
        self.num_channels = self.weights.shape[-1]
        self.activate_type = activate_type        
        self.is_normlize = is_normlize
        self.weight_updates = 0
        self.bias_updates = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        # batch normlize
        if self.is_normlize:
            self.BN_layer = BatchNormalizeLayer(self.num_channels, self.learning_rate, self.momentum)
    def forward(self, inputs, is_training=True):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights)
        if self.is_normlize:
            self.outputs = self.BN_layer.forward(self.outputs, is_training)
        else:
            self.outputs = self.outputs + self.biases
        self.outputs = activate(self.outputs, self.activate_type)
        return self.outputs
    def backward(self, delta):
        self.delta = delta * gradient(self.outputs, self.activate_type)
        if self.is_normlize:
            self.delta = self.BN_layer.backward(self.delta)
        else:
            self.bias_deltas = np.sum(self.delta, 0)
        self.weight_deltas = np.dot(self.inputs.T, self.delta)
        self.delta = np.dot(self.delta, self.weights.T)
        return self.delta
    def update(self):
        self.weight_updates = self.weight_updates * self.momentum + self.weight_deltas
        self.weights += self.learning_rate * self.weight_updates
        if self.is_normlize:
            self.BN_layer.update()
        else:
            self.bias_updates = self.bias_updates * self.momentum + self.bias_deltas
            self.biases += self.learning_rate * self.bias_updates

class ConvolutionLayer:

    def __init__(self, weights, biases, stride, activate_type=None, learning_rate=1e-3):
        self.weights = weights
        self.biases = biases
        self.ksize, _, self.in_channels, self.out_channels = self.weights.shape
        self.pad = self.ksize / 2
        self.stride = stride
        self.activate_type = activate_type        
        self.learning_rate = learning_rate
    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size, self.in_height, self.in_width, self.in_channels = self.inputs.shape
        self.out_height = (self.in_height + 2 * self.pad - self.ksize) / self.stride + 1
        self.out_width = (self.in_width + 2 * self.pad - self.ksize) / self.stride + 1
        self.out_shape = (self.batch_size, self.out_height, self.out_width, self.out_channels)
        self.weight_col = self.weights.reshape(self.ksize*self.ksize*self.in_channels, self.out_channels)
        input_col = im2col(self.inputs, self.batch_size, self.in_height, self.in_width, self.out_height, self.out_width, self.in_channels, self.ksize, self.stride, self.pad)
        self.outputs = (np.dot(input_col, self.weight_col)).reshape(self.out_shape)
        self.outputs = self.outputs + self.biases
        self.outputs = activate(self.outputs, self.activate_type)
        return self.outputs
    def backward(self, delta):
#        self.delta = delta.reshape(self.batch_size, self.out_height, self.out_width, self.out_channels)
        self.delta = delta * gradient(self.outputs, self.activate_type)
        self.bias_deltas = np.sum(self.delta, axis=(0,1,2))
        input_col = im2col(self.inputs, self.batch_size, self.in_height, self.in_width, self.out_height, self.out_width, self.in_channels, self.ksize, self.stride, self.pad)
        input_col_reshape = input_col.reshape(self.batch_size*self.out_height*self.out_width, self.ksize*self.ksize*self.in_channels)
        delta_reshape = self.delta.reshape(self.batch_size*self.out_height*self.out_width, self.out_channels)
        weight_delta_col = np.dot(input_col_reshape.T, delta_reshape)
        self.weight_deltas = weight_delta_col.reshape(self.ksize, self.ksize, self.in_channels, self.out_channels)
        delta_reshape = self.delta.reshape(self.batch_size,self.out_height*self.out_width, self.out_channels)
        delta_col = np.dot(delta_reshape, self.weight_col.T)
        self.delta =  col2im(delta_col, self.batch_size, self.in_height, self.in_width, self.out_height, self.out_width, self.in_channels, self.ksize, self.stride, self.pad)
        return self.delta
    def update(self):
        self.weights += self.learning_rate * self.weight_deltas
        self.biases += self.learning_rate * self.bias_deltas

class PoolLayer:

    def __init__(self, ksize, stride):
        self.ksize = ksize
        self.stride = stride
        self.pad = 0
        self.pad_value = float('inf')
        
    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size, self.in_height, self.in_width, self.in_channels = self.inputs.shape
        self.out_height = (self.in_height + 2 * self.pad - self.ksize) / self.stride + 1
        self.out_width = (self.in_width + 2 * self.pad - self.ksize) / self.stride + 1
        input_col = im2col(self.inputs, self.batch_size, self.in_height, self.in_width, self.out_height, self.out_width, self.in_channels, self.ksize, self.stride, self.pad, self.pad_value)
        
        input_col = input_col.reshape(self.batch_size, self.out_height*self.out_width, self.ksize*self.ksize, self.in_channels)
        input_col_max = np.max(input_col, axis = 2)
        self.input_col_argmax = np.argmax(input_col, axis = 2)
        self.outputs = input_col_max.reshape(self.batch_size, self.out_height, self.out_width, self.in_channels)
        return self.outputs
    def backward(self, delta):
        delta_col = np.zeros((self.batch_size, self.out_height*self.out_width, self.ksize*self.ksize, self.in_channels))
        delta_reshape = delta.reshape(self.batch_size, self.out_height*self.out_width, self.in_channels)
        for i in range(self.batch_size):
            for j in range(self.out_height*self.out_width):
                 for k in range(self.in_channels):
                     delta_col[i,j,self.input_col_argmax[i,j,k],k] = delta_reshape[i,j,k]
        delta_col = delta_col.reshape(self.batch_size, self.out_height*self.out_width, self.ksize*self.ksize*self.in_channels)
        self.delta = col2im(delta_col, self.batch_size, self.in_height, self.in_width, self.out_height, self.out_width, self.in_channels, self.ksize, self.stride, self.pad)
        return self.delta
        
class BatchNormalizeLayer:

    def __init__(self, num_channels, learning_rate, momentum):
        self.rolling_mean = 0
        self.rolling_variance = 0
        self.scales = np.ones(num_channels)
        self.biases = np.zeros(num_channels)
        self.bias_updates = 0
        self.scale_updates = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.EPS = 0.00001
    def forward(self, inputs, is_training):
        self.inputs = inputs
        self.axis = tuple(range(self.inputs.ndim - 1))
        if is_training:
            self.mean = self.inputs.mean(self.axis)
            self.variance = self.inputs.var(self.axis)
            self.rolling_mean = 0.99 * self.rolling_mean +  0.01 * self.mean
            self.rolling_variance = 0.99 * self.rolling_variance +  0.01 * self.variance
            self.norm = (self.inputs - self.mean) / np.sqrt(self.variance + self.EPS)
        else: 
            self.norm = (self.inputs - self.rolling_mean) / np.sqrt(self.rolling_variance + self.EPS)
        self.outputs = self.norm * self.scales + self.biases
        return self.outputs
    def backward(self, delta):
       self.delta = delta
       self.biase_deltas =  np.sum(self.delta, self.axis)
       self.scale_deltas = np.sum(self.norm * self.delta, self.axis)
       self.delta *= self.scales
       self.mean_delta = np.sum(self.delta, self.axis) * (-1.0) / np.sqrt(self.variance + self.EPS)
       self.variance_delta = np.sum(self.delta * (self.inputs-self.mean), self.axis) * (-0.5) * np.power(self.variance, -3.0/2)
       self.delta = self.delta * 1.0 / np.sqrt(self.variance+self.EPS) + self.variance_delta * 2 * (self.inputs-self.mean) / (self.inputs.size / self.inputs.shape[-1]) + self.mean_delta / (self.inputs.size / self.inputs.shape[-1])
       return self.delta
    def update(self):
        self.bias_updates = self.bias_updates * self.momentum + self.biase_deltas
        self.biases += self.learning_rate * self.bias_updates
        self.scale_updates = self.scale_updates * self.momentum + self.scale_deltas
        self.scales += self.learning_rate * self.scale_updates
        
class SoftmaxLayer:

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = softmax(self.inputs)
        return self.outputs
    def backward(self, targets):
        self.error = np.where(targets>0,-np.log(self.outputs),0)
        self.error = np.sum(self.error) / self.error.shape[0] 
        self.delta = targets-self.outputs 
        return self.delta, self.error

class DropoutLayer:

    def forward(self, inputs, dropout):
        self.inputs = inputs
        self.rand = np.random.rand(*self.inputs.shape) < dropout 
        return self.inputs * self.rand
  
    def backward(self, delta):
        self.delta = np.where(self.rand>0,delta,0)
        return self.delta

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
w1 = random_initializer([5, 5, 1, 16],-0.1, 0.1)
b1 = constant_initializer([16], 0.1)
w2 = random_initializer([5, 5, 16, 32],-0.1, 0.1)
b2 = constant_initializer([32], 0.1)
w3 = random_initializer([1568, 10], -0.1, 0.1)
b3 = constant_initializer([10], 0.1)
lr = 0.001
layer1 = ConvolutionLayer(w1,b1, 1,'relu')
layer2 = PoolLayer(2,2)
layer3 = ConvolutionLayer(w2,b2, 1,'relu')
layer4 = PoolLayer(2,2)
layer5 = ConnectedLayer(w3,b3,'', lr)
layer6 = SoftmaxLayer()

for step in range(2000):
    inputs, targets = mnist.train.next_batch(100)
    inputs =  inputs.reshape(100,28,28,1) 
    outputs = layer1.forward(inputs)
    outputs = layer2.forward(outputs)
    outputs = layer3.forward(outputs)
    outputs = layer4.forward(outputs)
    outputs = layer5.forward(outputs.reshape(100,-1))
    outputs = layer6.forward(outputs)
    delta, loss = layer6.backward(targets)
    delta = layer5.backward(delta)
    delta = layer4.backward(delta)
    delta = layer3.backward(delta)
    delta = layer2.backward(delta)
    delta = layer1.backward(delta)
    layer5.update()
    layer3.update()
    layer1.update()
    acc = np.mean((np.equal(np.argmax(outputs,1),np.argmax(targets,1))).astype(np.float))
    print loss,acc
'''
w1 = random_initializer([784,1024],-0.1, 0.1)
b1 = constant_initializer([1024], 0.1)
w2 = random_initializer([1024, 10], -0.1, 0.1)
b2 = constant_initializer([10], 0.1)
lr = 0.001

layer1 = ConnectedLayer(w1,b1,'relu', lr)
layer2_drop = DropoutLayer()
layer2 = ConnectedLayer(w2,b2, 'linear',lr)
layer3 = SoftmaxLayer()
total_loss = []
for step in range(2000):
    inputs, targets = mnist.train.next_batch(100)
    outputs = layer1.forward(inputs)
    outputs = layer2_drop.forward(outputs, 0.5)
    outputs = layer2.forward(outputs) 
    outputs = layer3.forward(outputs)
    delta, loss = layer3.backward(targets)
    delta = layer2.backward(delta)
    delta = layer2_drop.backward(delta)
    layer1.backward(delta)
    layer2.update()
    layer1.update()
    
    test_inputs,test_targets = mnist.test.next_batch(100) 
    test_outputs = layer1.forward(test_inputs, False)
    test_outputs = layer2.forward(test_outputs, False) 
    test_outputs = layer3.forward(test_outputs)
    acc = np.mean((np.equal(np.argmax(test_outputs,1),np.argmax(test_targets,1))).astype(np.float))
    
    print loss,acc
plt.plot(total_loss)
plt.legend()
plt.show()
'''   
    
