# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import struct
from MiniFramework.DataReader_2_0 import *
import matplotlib.pyplot as plt

class GeometryDataReader(DataReader_2_0):
    # mode = 'image' or 'vector' 
    def __init__(self, train_file, test_file, mode):
        super(GeometryDataReader, self).__init__(train_file, test_file) # ToDo: 需要查一下super的使用方法
        self.mode = mode

    def ConvertToGray(self, data):
        (N,C,H,W) = data.shape
        new_data = np.empty((N,H*W))
        # new_data = np.empty((N,C*H*W))        
        for i in range(N):
            if C == 3: # color
                new_data[i] = np.dot([0.299,0.587,0.114], data[i].reshape(3,-1)).reshape(1,784)
                # 根据文字部分的要求，直接使用2352(=784x3)的矢量做为样本特征值
                # new_data[i] = data[i].reshape(1, 2352) # 784 x 3 = 2352, 
            elif C == 1: # gray
                new_data[i] = data[i,0].reshape(1,784)
        #end if
        return new_data

    def NormalizeX(self):
        if self.mode == 'vector':
            XTrain = self.ConvertToGray(self.XTrainRaw)
            XTest = self.ConvertToGray(self.XTestRaw)
            self.XTrain = self.__NormalizeData(XTrain)
            self.XTest = self.__NormalizeData(XTest)
            self.num_feature = 784
            # self.num_feature = 2352
        else:
            self.XTrain = self.__NormalizeData(self.XTrainRaw)
            self.XTest = self.__NormalizeData(self.XTestRaw)

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
