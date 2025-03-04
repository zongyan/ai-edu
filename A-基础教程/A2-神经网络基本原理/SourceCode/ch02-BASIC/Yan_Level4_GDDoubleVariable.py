# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:05:48 2021

@author: yan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_function(x,y):
    J = x**2 + np.sin(y)**2
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x, 2*np.sin(y)*np.cos(y)])

def show_3d_surface(x, y, z):
    # the following two are used to generate the 3d-plot axis
    fig = plt.figure()
    ax = Axes3D(fig)
 
    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v) # genearte the mesh grid (x and y axises)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j]**2 + np.sin(Y[i, j])**2 # calculate the z-axis value

    ax.plot_surface(X, Y, R, cmap='rainbow') # plot the 3d surface version
    plt.plot(x,y,z,c='black') # plot the 3d figure
    plt.show()
    
if __name__ == '__main__':
    theta = np.array([3, 1])
    eta = 0.1 # learning speed
    error = 1e-2
    X = []
    Y = []
    Z = []
    
    for i in range(100):
        print(theta)
        x=theta[0]
        y=theta[1]
        z=target_function(x, y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("x=%f, y=%f, z=%f"%(x,y,z))
        d_delta=derivative_function(theta)
        theta = theta-eta*d_delta
        if z < error:
            break
        
    show_3d_surface(X, Y, Z)

    

