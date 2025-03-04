# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy
import numba
import time

from tqdm import tqdm
from MiniFramework.ConvWeightsBias import *
from MiniFramework.ConvLayer import *
from MiniFramework.HyperParameters_4_2 import *

def conv_4d(x, weights, bias, out_h, out_w, stride=1):
    # 输入图片的批大小，通道数，高，宽
    assert(x.ndim == 4)
    # 输入图片的通道数
    assert(x.shape[1] == weights.shape[1])  
    batch_size = x.shape[0]
    num_input_channel = x.shape[1]
    num_output_channel = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]
    rs = np.zeros((batch_size, num_output_channel, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(num_output_channel):
            rs[bs,oc] += bias[oc]
            # xx = np.array([3,2,1,0]).reshape(1,1,2,2) # get a 4-D data: [[[[3 2] [1 0]]]]
            # yy = xx[0,0,1] + xx[0,0,0] # sum in the 3-d dimension: [4 2], xx[0,0,1]直接读取相应维度的数据，又是因为xx本身是思维的，所以读取出来就是变成了一维的
            for ic in range(num_input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,fh+ii,fw+jj] * weights[oc,ic,fh,fw]
    return rs

def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)

def test_2d_conv():
    batch_size = 1
    stride = 1
    padding = 0
    fh = 2
    fw = 2
    input_channel = 1
    output_channel = 1
    iw = 3
    ih = 3
    (output_height, output_width) = calculate_output_size(ih, iw, fh, fw, padding, stride)
    wb = ConvWeightsBias(output_channel, input_channel, fh, fw, InitialMethod.MSRA, OptimizerName.SGD, 0.1)
    wb.Initialize("test", "test", True)
    wb.W = np.array([3,2,1,0]).reshape(1,1,2,2) 
    # D1: 1; D2: 1; D3: 2; D4: 2; 这个就是重新安排维度，即第一维是1个数据；第二维是
    # 1个数据；第三维是2个数据，第四维是2个数据
    wb.B = np.array([0])
    x = np.array(range(9)).reshape(1,1,3,3)
    # Solution 1 in Section 17.2
    output0 = conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    print("input=\n", x)
    print("weights=\n", wb.W)
    print("output=\n", output0)    
    
    # Solution 2 in Section 17.2
    output1 = jit_conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    print("input=\n", x)
    print("weights=\n", wb.W)
    print("output=\n", output1)

    # Solution 3 in Section 17.2
    col = img2col(x, 2, 2, 1, 0)
    w = wb.W.reshape(4, 1)
    output2 = np.dot(col, w)
    print("input=\n", col)
    print("weights=\n", w)
    print("output2=\n", output2)


def test_4d_im2col():
    batch_size = 2
    stride = 1
    padding = 0
    fh = 2
    fw = 2
    input_channel = 3
    output_channel = 2
    iw = 3
    ih = 3

    x = np.random.randn(batch_size, input_channel, iw, ih)
    params = HyperParameters_4_2(
        0.1, 1, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)
    c1 = ConvLayer((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c1.initialize("test", "test", False)
    f1 = c1.forward_numba(x)
    f2 = c1.forward_img2col(x)
    print("correctness:", np.allclose(f1, f2, atol=1e-7))

def understand_4d_im2col():
    batch_size = 2
    stride = 1
    padding = 0
    fh = 2
    fw = 2
    input_channel = 3
    output_channel = 2
    iw = 3
    ih = 3
    (output_height, output_width) = calculate_output_size(ih, iw, fh, fw, padding, stride)
    wb = ConvWeightsBias(output_channel, input_channel, fh, fw, InitialMethod.MSRA, OptimizerName.SGD, 0.1)
    wb.Initialize("test", "test", True)
    wb.W = np.array(range(output_channel * input_channel * fh * fw)).reshape(output_channel, input_channel, fh, fw)
    wb.B = np.array([0])
    x = np.array(range(input_channel * iw * ih * batch_size)).reshape(batch_size, input_channel, ih, iw)

    col = img2col(x, 2, 2, 1, 0)
    w = wb.W.reshape(output_channel, -1).T
    output = np.dot(col, w)
    print("x=\n", x)
    print("col_x=\n", col)
    print("weights=\n", wb.W)
    print("col_w=\n", w)
    print("output=\n", output)
    out2 = output.reshape(batch_size, output_height, output_width, -1)
    print("out2=\n", out2)
    out3 = np.transpose(out2, axes=(0, 3, 1, 2)) # 参见Section 17.2的Solution 3文字部分
    print("conv result=\n", out3)
    """
    reshape的作用就是重新配置array的维度，transpose的作用则是可以调换array特定维度
    的数据    
    """

def test_performance():
    batch_size = 64
    params = HyperParameters_4_2(
        0.1, 1, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)
    stride = 1
    padding = 1
    fh = 3
    fw = 3
    input_channel = 3
    output_channel = 4
    iw = 28
    ih = 28
    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    x = np.random.randn(batch_size, input_channel, iw, ih)
    
    c1 = ConvLayer((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c1.initialize("test", "test", False)

    # dry run
    for i in tqdm(range(5)):
        f1 = c1.forward_numba(x)
        delta_in = np.ones((f1.shape))
        #b1, dw1, db1 = c1.backward_numba(delta_in, 1)
    # run
    s1 = time.time()
    for i in tqdm(range(1000)):
        f1 = c1.forward_numba(x)
        #b1, dw1, db1 = c1.backward_numba(delta_in, 1)
    e1 = time.time()
    print("method numba:", e1-s1)

    # dry run
    for i in tqdm(range(5)):
        f2 = c1.forward_img2col(x)
        #b2, dw2, db2 = c1.backward_col2img(delta_in, 1)
    # run
    s2 = time.time()
    for i in tqdm(range(1000)):
        f2 = c1.forward_img2col(x)
        #b2, dw2, db2 = c1.backward_col2img(delta_in, 1)
    e2 = time.time()
    print("method img2col:", e2-s2)

    print("compare correctness of method 1 and method 2:")
    print("forward:", np.allclose(f1, f2, atol=1e-7))
    #print("backward:", np.allclose(b1, b2, atol=1e-7))
    #print("dW:", np.allclose(dw1, dw2, atol=1e-7))
    #print("dB:", np.allclose(db1, db2, atol=1e-7))

if __name__ == '__main__':
    test_2d_conv()
    understand_4d_im2col()
    test_4d_im2col()
    test_performance()
