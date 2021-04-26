# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_1 import *

file_name = "../../SourceCode/Data/ch05.npz"

# main
if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX() # input data is two columns, need to normalise
    reader.NormalizeY() # output data is one column, need to normalise
    # net
    hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=200, batch_size=10, eps=1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    debug_x = np.array([x1,x2]) # debug_x is a matrix (1x2)
    print(f"debug_x is equal to {debug_x}")
    print(f"debug_x.size is equal to {debug_x.size}")
    debug_x = np.array([x1,x2]).reshape(1,2) # debug_x is an matrix (2x1)
    print(f"debug_x is equal to {debug_x}")
    print(f"debug_x.size is equal to {debug_x.size}")    
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("z=", z)
    Z_true = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z_true=", Z_true)
