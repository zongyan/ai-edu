# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.DataReader_1_1 import *

file_name = "../../SourceCode/Data/ch05.npz"

if __name__ == '__main__':
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0]
    """
    The following shows the difference of shape in column and row matrices
    mm = np.array([1, 2, 3, 4, 5]) # this is a row matrix
    print(mm.shape) # (5,)
    nn = np.array([[1], [2], [3], [4], [5]]) # this is a column matrix
    print(nn.shape) # (5, 1)
    
    This means that when useing the .shape[0], I need print all infos of .shape,
    and then determing which one I need to use (i.e. .shape[0] or shape[1])
    """
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:]))) # 合并成为新矩阵
                                                     # the difference between 
                                                     # the numpy.column_stack 
                                                     # and numpy.row_stack
    
    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    """
    The following link present a liitle more details about the differences between 
    array and matrix: 
    https://blog.csdn.net/wkl7123/article/details/84332899?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control
    
    But, more study about these two are required
    """
    b = np.asmatrix(a) # numpy.asmatrix & numpy.asarray can be used to 
                       # do the exchange between array and matrix
    c = np.linalg.inv(b) # compute the (multiplicative) inverse of a matrix
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)
    # inference
    z = w1 * 15 + w2 * 93 + b
    print("z=",z)
