# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# this file is to generate the data for use in Chapter 5

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

file_name = "./ch05.npz"
# file_name = "../../SourceCode/Data/ch05.npz" 

"""
both of these two path can be used. 
对于两者的区别， 主要在于前者是基于current directory下的ch05.npz文件， 对于current 
directory的定义如下：
When you run a Python script, the current working directory is set to the 
directory from which the script is executed.

后者就是基于current directory上，往上走两级，然后接着就是又回到SourceCode/Data的
directory下面。（这个也是根据本script最后三行代码，即65-67行）
"""

def TargetFunction(x1,x2):
    w1,w2,b = 2,5,10
    return w1*(20-x1) + w2*x2 + b

def CreateSampleData(m):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        X = data["data"]
        Y = data["label"]
    else:
        X = np.zeros((m,2))
        
        # radius [2,20]
        X[:,0:1] = (np.random.random(1000)*20+2).reshape(1000,1)
        debug_x = np.random.random(1000) # return the next random floating point 
                                         # number in the range [0.0, 1.0)                                         
        print(debug_x) 
        print(debug_x.shape) # size -> (1000,)
        debug_x = (np.random.random(1000)*20+2).reshape(1000,1)
        print(debug_x) # size -> (1000,1)
        print(debug_x.shape) # size -> (1000,)        
        
        # [40,120] square
        X[:,1:2] = np.random.randint(40,120,(m,1)) # return a random integer 
                                                   # N such that a <= N <= b
        Y = TargetFunction(X[:,0:1], X[:,1:2])
        Noise = np.random.randint(1,100,(m,1)) - 50
        Y = Y + Noise
        np.savez(file_name, data=X, label=Y)
    return X, Y

if __name__ == '__main__':
    X,Y = CreateSampleData(1000)

    print(X[:,0].max()) # return the max number in column[0]
    print(X[:,0].min()) # return the min number in column[0]
    print(X[:,0].mean()) # return the mean number in column[0]
    
    print(X[:,1].max()) # return the max number in column[1]
    print(X[:,1].min()) # return the min number in column[1]
    print(X[:,1].mean()) # return the mean number in column[1]
    
    print(Y.max())
    print(Y.min())
    print(Y.mean())

    fig = plt.figure() # create a new figure, or activate an existing figure
    ax = Axes3D(fig) # 3D axes object
    ax.scatter(X[:,0],X[:,1],Y)
    plt.show()
    
    print(os.getcwd()) # show the current working directory 
    os.chdir("../..") # change the parent folder, and the parent folder again.
    print(os.getcwd())  # show the current working directory again