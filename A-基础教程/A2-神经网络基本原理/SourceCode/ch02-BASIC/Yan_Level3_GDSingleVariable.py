# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:05:48 2021

@author: yan
"""

import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    y = x*x
    return y

def derivative_function(x):
    return 2*x

def draw_function():
    x = np.linspace(-1.2, 1.2)
    y = target_function(x)
    plt.plot(x,y)
    
def draw_gd(X):
    Y=[]
    for i in range(len(X)):
        Y.append(target_function(X[i]))
        
    plt.plot(X, Y)
    
if __name__ == '__main__':
    x = 1.2
    eta = 0.3 # learning speed
    error = 1e-3
    X = []
    X.append(x)
    y=target_function(x)
    while y > error:
        x = x - eta * derivative_function(x)
        X.append(x)
        y = target_function(x)
        print("x=%f, y=%f"%(x, y))
        
    draw_function()
    draw_gd(X)
    plt.show()  # I still YET understand this line!


    

