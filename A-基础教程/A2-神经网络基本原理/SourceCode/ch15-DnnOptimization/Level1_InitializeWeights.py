# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_1 import *
from MiniFramework.ActivationLayer import *

def net(init_method, activator):

    max_epoch = 1
    batch_size = 5
    learning_rate = 0.02

    params = HyperParameters_4_1(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=init_method)

    net = NeuralNet_4_1(params, "level1")
    num_hidden = [128,128,128,128,128,128,128]
    fc_count = len(num_hidden)-1
    layers = []

    """
    这一种方式就是有点类似于我现在写matlab代码的方式了，如果会看之前的tutorial代码，
    就是可以明白有多少层hidden layer，就是会写多少次的了。这会儿就是变了，就是使用
    一个for循环，大大节省了工作量了，也是简化的代码。
    """
    for i in range(fc_count):
        fc = FcLayer_1_1(num_hidden[i], num_hidden[i+1], params)
        net.add_layer(fc, "fc")
        layers.append(fc)

        ac = ActivationLayer(activator)
        net.add_layer(ac, "activator")
        layers.append(ac)
    # end for
    
    # 从正态分布中取1000个样本，每个样本有num_hidden[0]个特征值
    # 转置是为了可以和w1做矩阵乘法
    x = np.random.randn(1000, num_hidden[0])

    # 激活函数输出值矩阵列表
    a_value = []

    """
    下面的两个for循环，就是给出怎么样子把这个每一层的结果输出保存下来，然后就是plot
    出来的方式了。其实Section 15.1的第二个小问题就是可以按照这个方式做就可以了。
    使用hist这个函数就是可以画出histgram，然后flatten的作用是把数据拉平的。
    
    这里就是有一点需要注意一下的了，这个就是对于这个output的尺寸，他是和样本数据，以
    及每一层的neurons的数量有关系的。同时，需要注意的是，这个是没有训练的神经网络，
    只是使用了不同的初始化方式，然后查看每一层的输出结果。
    
    不过Section 14.6的代码由于数据的问题，一直是没有办法运行起来的了。所以就暂时没有
    上手实际操作的了。    
    """
    # 依次做所有层的前向计算
    input = x
    for i in range(len(layers)):
        output = layers[i].forward(input)
        # 但是只记录激活层的输出
        if isinstance(layers[i], ActivationLayer):
            a_value.append(output)
        # end if
        input = output
    # end for

    for i in range(len(a_value)):
        ax = plt.subplot(1, fc_count+1, i+1)
        ax.set_title("layer" + str(i+1))
        plt.ylim(0,10000)
        if i > 0:
            plt.yticks([])
        ax.hist(a_value[i].flatten(), bins=25, range=[0,1])
    #end for
    # super title
    plt.suptitle(init_method.name + " : " + activator.get_name())
    plt.show()

if __name__ == '__main__':
    net(InitialMethod.Normal, Sigmoid())
    net(InitialMethod.Xavier, Sigmoid())
    net(InitialMethod.Xavier, Relu())
    net(InitialMethod.MSRA, Relu())
