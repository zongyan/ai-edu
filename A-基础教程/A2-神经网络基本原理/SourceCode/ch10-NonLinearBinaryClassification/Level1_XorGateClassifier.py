# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from HelperClass2.NeuralNet_2_1 import *

# x1=0,x2=0,y=0
# x1=0,x2=1,y=1
# x1=1,x2=0,y=1
# x1=1,x2=1,y=0
class XOR_DataReader(DataReader_2_0):
    def __init__(self):
        pass # The pass statement is used as a placeholder for future code.
             # When the pass statement is executed, nothing happens, but you 
             # avoid getting an error when empty code is not allowed.
             # Empty code is not allowed in loops, function definitions, class 
             # definitions, or in if statements.

    def ReadData(self):
        self.XTrain = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        self.YTrain = np.array([0,1,1,0]).reshape(4,1)

        self.num_category = 1
        self.num_train = self.XTrain.shape[0]
        print(f"debug: self.num_train is {self.num_train}") # --> 4
        self.num_feature = self.XTrain.shape[1]
        print(f"debug: self.num_feature is {self.num_feature}") # --> 2

        self.XTest = self.XTrain
        self.YTest = self.YTrain
        self.XDev = self.XTrain
        self.YDev = self.YTrain        
        self.num_test = self.num_train


def Test(dataReader, net):
    print("testing...")
    X,Y = dataReader.GetTestSet()
    A2 = net.inference(X)
    print("A2=",A2)
    diff = np.abs(A2-Y)
    result = np.where(diff < 1e-2, True, False)
    """
    a = np.array([[0, 1, 2],
                  [0, 2, 4],
                  [0, 3, 6]])
    np.where(a < 4, a, -1)  # -1 is broadcast
    results are ----->
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])    
    """
    if result.sum() == dataReader.num_test:
        return True
    else:
        return False

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()

    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005

    hp = HyperParameters_2_0(
        n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, 
        NetType.BinaryClassifier, 
        InitialMethod.Xavier)
    net = NeuralNet_2_1(hp, "Xor_221")

    net.train(dataReader, 100, True)
    net.ShowTrainingHistory()

    print(Test(dataReader, net))
