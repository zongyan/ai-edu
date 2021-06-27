# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.jit_utility import *

class PoolingLayer(CLayer):
    def __init__(self,
                input_shape,    # (input_c, input_h, input_w)
                pool_shape,     # (pool_h, pool_w)
                stride, 
                pooling_type):  # MAX, MEAN
        self.num_input_channel = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.pool_height = pool_shape[0]
        self.pool_width = pool_shape[1]
        self.stride = stride
        self.pooling_type = pooling_type

        self.pool_size = self.pool_height * self.pool_width
        self.output_height = (self.input_height - self.pool_height) // self.stride + 1
        self.output_width = (self.input_width - self.pool_width) // self.stride + 1
        self.output_shape = (self.num_input_channel, self.output_height, self.output_width)
        self.output_size = self.num_input_channel * self.output_height * self.output_width
        
        self.x = None
        self.arg_max = None

    def initialize(self, folder, name):
        self.init_file_name = str.format("{0}/{1}_init.npy", folder, name)

    def forward(self, x, train=True):
        return self.forward_numba(x, train)

    def backward(self, delta_in, layer_idx):
        return self.backward_numba(delta_in, layer_idx)

    def forward_img2col(self, x, train=True):
        self.x = x
        N, C, H, W = x.shape
        col = img2col(x, self.pool_height, self.pool_width, self.stride, 0)
        col_x = col.reshape(-1, self.pool_height * self.pool_width) # reshape的代码需要专门研究一下
        self.arg_max = np.argmax(col_x, axis=1) # Returns the indices of the maximum values along an axis.
        out1 = np.max(col_x, axis=1)
        out2 = out1.reshape(N, self.output_height, self.output_width, C)
        self.z = np.transpose(out2, axes=(0,3,1,2))
        return self.z

    def backward_col2img(self, delta_in, layer_idx):
        dout = np.transpose(delta_in, (0,2,3,1))
        dmax = np.zeros((dout.size, self.pool_size)).astype('float32')
        #dmax[np.arange(self.arg_max.size), np.flatten(self.arg_max)] = np.flatten(dout)
        # np.arange(self.arg_max.size)的作用是在0到self.arg_max.size，间隔1增加
        # 然后左边方括号里面的作用，就是分别是每一个row找到最大值的位置
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # Return a copy of the array collapsed into one dimension.
        dmax = dmax.reshape(dout.shape + (self.pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2img(dcol, self.x.shape, self.pool_height, self.pool_width, self.stride, 0, self.output_height, self.output_width)
        return dx

    def forward_numba(self, x, train=True):
        assert(x.ndim == 4)
        self.x = x
        self.batch_size = self.x.shape[0]
        self.z = jit_maxpool_forward(self.x, self.batch_size, self.num_input_channel, self.output_height, self.output_width, self.pool_height, self.pool_width, self.stride)
        return self.z

    def backward_numba(self, delta_in, layer_idx):
        assert(delta_in.ndim == 4)
        assert(delta_in.shape == self.z.shape)
        delta_out = jit_maxpool_backward(self.x, delta_in, self.batch_size, self.num_input_channel, self.output_height, self.output_width, self.pool_height, self.pool_width, self.stride)
        return delta_out

    def save_parameters(self):
        np.save(self.init_file_name, self.pooling_type)

    def load_parameters(self):
        self.mode = np.load(self.init_file_name, allow_pickle=True)
        pass
