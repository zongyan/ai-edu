# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from Activators.Relu import *
from Activators.Elu import *
from Activators.LeakyRelu import *
from Activators.Sigmoid import *
from Activators.Softplus import *
from Activators.Step import *
from Activators.Tanh import *
from Activators.BenIdentity import *

def Draw(start,end,func,lable1,lable2):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    da, dz = func.backward(z, a, 1) # dz暂时没有用到，Section 8.1中也是没有提到

    p1, = plt.plot(z,a)
    p2, = plt.plot(z,da)
    plt.legend([p1,p2], [lable1, lable2]) # 这个legend的方式比我的复杂一些，和我
                                          # 的不同在于，这个方式就是把多了legend
                                          # 放到一起了，这一点比我的好
    plt.grid()
    plt.xlabel("input : z")
    plt.ylabel("output : a")
    plt.title(lable1)
    plt.show()

if __name__ == '__main__':
    Draw(-7,7,CSigmoid(),"Sigmoid Function","Derivative of Sigmoid")
    Draw(-7,7,CSigmoid(),"Logistic Function","Derivative of Logistic")
    Draw(-5,5,CStep(0.3),"Step Function","Derivative of Step")
