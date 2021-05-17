# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass2.NeuralNet_3_0 import *

# Roll all our parameters dictionary into a single vector satisfying our specific required shape.
def dictionary_to_vector(dict_params):
    keys = []
    count = 0
    for key in ["W1", "B1", "W2", "B2", "W3", "B3"]:
        
        # flatten parameter
        new_vector = np.reshape(dict_params[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]   # -> ["W1","W1",..."b1","b1",..."W2"...]
        
        if count == 0:
            theta = new_vector
        else:         #np.concatenate
            theta = np.concatenate((theta, new_vector), axis=0) 
            # theta的结构是一个Nx1的向量，然后向量的结构是按照如下的顺序
            # W1, B1, W2, B2, W3, B3， 对于W1这样的矩阵，就是会每一行一行的进行
            # 转换成向量的形式
        count = count + 1
 
    return theta, keys

# roll all grad values into one vector, the same shape as dictionary_to_vector()
# 主要思路和dictionary_to_vector()是一样的
def gradients_to_vector(gradients):
    count = 0
    for key in ["dW1", "dB1", "dW2", "dB2", "dW3", "dB3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
       
        if count == 0:
            d_theta = new_vector
        else:
            d_theta = np.concatenate((d_theta, new_vector), axis=0)
        count = count + 1
 
    return d_theta

# Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
def vector_to_dictionary(theta, layer_dims):
    dict_params = {}
    L = 4  # the number of layers in the networt
    start = 0
    end = 0
    for l in range(1,L):
        end += layer_dims[l]*layer_dims[l-1]
        dict_params["W" + str(l)] = theta[start:end].reshape((layer_dims[l-1],layer_dims[l]))
        start = end
        end += layer_dims[l]*1
        dict_params["B" + str(l)] = theta[start:end].reshape((1,layer_dims[l]))
        start = end
    #end for
    return dict_params

# cross entropy: -Y*lnA
def CalculateLoss(net, dict_Param, X, Y, count, ):
    net.wb1.W = dict_Param["W1"]
    net.wb1.B = dict_Param["B1"]
    net.wb2.W = dict_Param["W2"]
    net.wb2.B = dict_Param["B2"]
    net.wb3.W = dict_Param["W3"]
    net.wb3.B = dict_Param["B3"]
    net.forward(X)
    p = Y * np.log(net.output)
    Loss = -np.sum(p) / count
    return Loss


if __name__ == '__main__':

    n_input = 7
    n_hidden1 = 16
    n_hidden2 = 12
    n_output = 10
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters_3_0(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_3_0(hp, "MNIST_gradient_check")
    dict_Param = {"W1": net.wb1.W, "B1": net.wb1.B, "W2": net.wb2.W, "B2": net.wb2.B, "W3": net.wb3.W, "B3": net.wb3.B}
    '''
    input layer: 7; hidden layer 1: 16; hidden layer 2: 12; output layer: 10
    这里的说，都是可以理解为每一层neuron的个数（对于输入层有点不准确，但主要是为了
    表达清楚概念）。
    每一层有多少个neuron，就是有多少个bias B的了，所以，就是对于dict_Param来说，
    B1， B2， B3 (对应hidden & output layer)的维度就是16， 12， 10。
    但是呢，对于weights W的理解，就是不能够像这么简单的了，具体点说，
    对于hidden layer 1的weights来说，input layer的第一个neuron就是会和hidden layer 1
    的所有neurons相连接（1 x 16），input layer的第二个neuron就是会和hidden layer 1
    的所有neurons相连接（1 x 16），以此类推，所以，W1的维度是7x16， W2的维度是16x12， 
    W3的维度是12x10
    '''
    layer_dims = [n_input, n_hidden1, n_hidden2, n_output]
    n_example = 2
    x = np.random.randn(n_example, n_input)
    #y = np.array([1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,1]).reshape(-1,n_example)
    #y = np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]).reshape(-1,n_example)
    y = np.array([1,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
    
    net.forward(x)
    net.backward(x, y)
    dict_Grads = {"dW1": net.wb1.dW, "dB1": net.wb1.dB, "dW2": net.wb2.dW, "dB2": net.wb2.dB, "dW3": net.wb3.dW, "dB3": net.wb3.dB}
    # dict_Grads 和 dict_Param的维度是一样的，参见上面的解释。
    
    """
    因为W和B是矩阵，而不是一个标量，需要先是向量化 
    """    
    J_theta, keys = dictionary_to_vector(dict_Param)
    d_theta_real = gradients_to_vector(dict_Grads)

    n = J_theta.shape[0]
    J_plus = np.zeros((n,1))
    J_minus = np.zeros((n,1))
    d_theta_approx = np.zeros((n,1))

    # for each of the all parameters in w,b array
    for i in range(n):
        J_theta_plus = np.copy(J_theta)
        J_theta_plus[i][0] = J_theta[i][0] + eps
        # 多分类交叉熵
        """
        这里就是需要使用更新之后的weights & biases，计算这个loss的数值。这个理解是
        正确的，就是加上一个数值eps（或者减去一个数值），得到更新之后的weights & biases，
        然后这个更新后的weights & biases来计算loss function的数值。 
        
        每一次增加或减去eta，都是仅仅对model中的一个weight进行的，然后经过range(n)
        次之后，就是得到了一个完成的d_theta_approx了。
        
        
        如图12.4，纵轴y就是每一次得到的loss function，而横轴，仅仅是model中的一个
        weight或者bias（x轴不是所有的参数，而仅仅是NN中的一个参数，比如说，如果网络
        中有N个weights & biases，那么图12.4就是应该是需要N张图）。
        """
        J_plus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_plus, layer_dims), x, y, n_example)

        J_theta_minus = np.copy(J_theta)
        J_theta_minus[i][0] = J_theta[i][0] - eps
        J_minus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_minus, layer_dims), x, y, n_example)

        d_theta_approx[i] = (J_plus[i] - J_minus[i]) / (2 * eps)
    # end for
    numerator = np.linalg.norm(d_theta_real - d_theta_approx)  ####np.linalg.norm 二范数
    denominator = np.linalg.norm(d_theta_approx) + np.linalg.norm(d_theta_real)
    difference = numerator / denominator
    print('diference ={}'.format(difference))
    if difference<1e-7:
        print("NO mistake.")
    elif difference<1e-4:
        print("Acceptable, but a little bit high.")
    elif difference<1e-2:
        print("May has a mistake, you need check code!")
    else:
        print("HAS A MISTAKE!!!")
    

