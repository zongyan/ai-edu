# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *

train_file = "../../SourceCode/Data/ch14.Income.train.npz"
test_file = "../../SourceCode/Data/ch14.Income.train.npz"
# test_file = "../../SourceCode/Data/ch14.Income.test.npz" 
# 在处理test数据的时候，没有办法把这个数据处理完成（出现bug），所以我就是会使用这个train
# 数据来代替test数据，主要也是为了过一遍代码的需要，并不是真正的需要测试这个trained
# model的性能

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

def model(dr):
    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_hidden4 = 4
    num_output = 1

    max_epoch = 100
    batch_size = 16
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopDiff, 1e-3))

    net = NeuralNet_4_0(params, "Income")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    a1 = ActivationLayer(Relu())
    net.add_layer(a1, "relu1")
    
    fc2 = FcLayer_1_0(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    a2 = ActivationLayer(Relu())
    net.add_layer(a2, "relu2")

    fc3 = FcLayer_1_0(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    a3 = ActivationLayer(Relu())
    net.add_layer(a3, "relu3")

    fc4 = FcLayer_1_0(num_hidden3, num_hidden4, params)
    net.add_layer(fc4, "fc4")
    a4 = ActivationLayer(Relu())
    net.add_layer(a4, "relu4")

    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    net.train(dr, checkpoint=1, need_test=True)
    return net
    
if __name__ == '__main__':
    dr = LoadData()
    net = model(dr)
    net.ShowLossHistory()
