# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
import cv2

from MiniFramework.ConvWeightsBias import *
from MiniFramework.ConvLayer import *
from MiniFramework.ActivationLayer import *
from MiniFramework.PoolingLayer import *
from MiniFramework.HyperParameters_4_2 import *
from MiniFramework.jit_utility import *

circle_pic = "circle.png"

def normalize(x, max_value=1):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)*max_value
    return x_n

def try_filters(file_name):
    img = cv2.imread(file_name) # loading an image: the decoded image has the 
                                # channels stored in B G R order, 80 x 80 pixels
    # cv2 format is:G B R, change it to R G B
    img1=img[:,:,[2,1,0]]    
    """   
    a1D = np.array([1, 2, 3, 4]) # 一维矩阵
    print(a1D)
    a2D = np.array([[1, 2], [3, 4]]) # 二维矩阵
    print(a2D)
    a3D = np.array([ [[1, 2], [3, 4]], [[5, 6], [7, 8]] ]) # 三维矩阵
    print(a3D)
    
    a3D1=a3D[:,:,:] # 保持原样
    print(a3D1)
    a3D2=a3D[:,:,[0,1]] # 保持原样
    print(a3D2)
    a3D3=a3D[:,:,[1,0]] # 第三维度的第0列；第1列的数据调换    
    print(a3D3) 
    理解高纬度（比如上面的第三维度）的数据的时候，不能够只是通过这个代码中的中括号；
    而是需要理解成Section 17.1的图17.16的样式（此时就是对列数据进行调换的时候，就
    变得形象深度了）。
    """
    #plt.imshow(img2)
    #plt.show()
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # convert from rgb to gray
    # 从变量的维度来看，img2是二维，也就是说最后一维是控制rgb的，或者说是控制彩或灰色
    batch_size = 1
    input_channel = 1
    (height, width) = img2.shape # 这一种赋值方式还是很好的， cv2.cvtColor只是把第三维度给简化了，维度还是和img一样的（高度x宽度）
    FH = 3 # kernal heigh
    FW = 3 # kernal width
    print(img2.shape)
    data = img2.reshape((1,1,height,width)) # 4-dimension array
    hp = HyperParameters_4_2(
        0.1, 10, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)
    conv = ConvLayer((1,height,width), (1,FH,FW), (1,1), hp)
    conv.initialize("know_cnn", "name")
    
    filters = [
        np.array([0,-1,0,
                  -1,5,-1,
                  0,-1,0]),         # sharpness filter
        np.array([0,0,0,
                  -1,2,-1,
                  0,0,0]),          # vertical edge
        np.array([1,1,1,
                  1,-9,1,
                  1,1,1]),          # surround
        np.array([-1,-2,-1,
                  0,0,0,
                  1,2,1]),          # sobel y
        np.array([0,0,0,
                  0,1,0,
                  0,0,0]),          # nothing
        np.array([0,-1,0,
                  0,2,0,
                  0,-1,0]),         # horizontal edge
        np.array([0.11,0.11,0.11,
                  0.11,0.11,0.11,
                  0.11,0.11,0.11]), # blur
        np.array([-1,0,1,
                  -2,0,2,
                  -1,0,1]),         # sobel x
        np.array([2,0,0,
                  0,-1,0,
                  0,0,-1])]         # embossing

    filters_name = ["sharpness", "vertical edge", "surround", "sobel y", "nothing", "horizontal edge", "blur", "sobel x", "embossing"]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
    for i in range(len(filters)):
        filter = np.repeat(filters[i], input_channel).reshape(batch_size, input_channel,FH,FW) 
        # numpy.repreat在input_channel=1的时候没有任何意义（因为这个function里面
        # 是单通道的数据），但是在input_channel > 1的时候，就是很有意义的了，因为此时
        # 就是三通道了， see line 133
        conv.set_filter(filter, None)
        z = conv.forward(data)
        #z = normalize(z, 255)
        ax[i//3, i%3].imshow(z[0,0]) # 从第一&二维度的位置0，0读取出一个数据
        ax[i//3, i%3].set_title(filters_name[i])
        ax[i//3, i%3].axis("off")
    plt.suptitle("filters")
    plt.show()
    return z

def conv_relu_pool():
    img = cv2.imread(circle_pic)
    #img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    batch_size = 1
    (height, width, input_channel) = img.shape
    FH = 3
    FW = 3
    data = np.transpose(img, axes=(2,1,0)).reshape((batch_size,input_channel,width,height)) 
    # np.transpose(img, axes=(2,1,0))的作用，就是把img从(80, 80, 3)的维度变成
    # (3, 80, 80)的尺度，这个尺度才是正常的维度；(80, 80, 3)是因为cv2.imread读取的
    # 数据方向是反方向的。
    hp = HyperParameters_4_2(
        0.1, 10, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)
    conv = ConvLayer((input_channel,width,height),(1,FH,FW),(1,0),hp)
    conv.initialize("know_cnn", "conv")
    kernal = np.array([ -1,0,1,
                        -2,0,2,
                        -1,0,1])
    filter = np.repeat(kernal, input_channel).reshape(batch_size, input_channel,FH,FW) # 每一个kernal中的元素都是会被重复input_channel的次数
    conv.set_filter(filter, None)
    z1 = conv.forward(data)
    z2 = Relu().forward(z1)
    pool = PoolingLayer(z2[0].shape, (2,2), 2, PoolingTypes.MAX)
    pool.initialize("know_cnn", "pool")
    z3 = pool.forward(z2)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    ax[0,0].imshow(img[:,:,[2,1,0]]) # “显示格式”和“读取的格式”，以及“数据处理的格式”都不一样，注意区分
    ax[0,0].axis("off")
    ax[0,0].set_title("source:" + str(img.shape))
    ax[0,1].imshow(z1[0,0].T) # 
    ax[0,1].axis("off")
    ax[0,1].set_title("conv:" + str(z1.shape))
    ax[1,0].imshow(z2[0,0].T)
    ax[1,0].axis("off")
    ax[1,0].set_title("relu:" + str(z2.shape))
    ax[1,1].imshow(z3[0,0].T)
    ax[1,1].axis("off")
    ax[1,1].set_title("pooling:" + str(z3.shape))

    plt.suptitle("conv-relu-pool")
    plt.show()

    
if __name__ == '__main__':
    try_filters(circle_pic)
    conv_relu_pool()
