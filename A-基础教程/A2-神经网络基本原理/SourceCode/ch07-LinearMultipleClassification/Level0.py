# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:19:33 2021

@author: Yan Zong
"""
import numpy as np

class Softmax(object):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        print(f"debug: np.max(z, axis=1, keepdims=True) is {np.max(z, axis=1, keepdims=True)}")
        print(f"debug: shift_z is {shift_z}")
        exp_z = np.exp(shift_z)
        print(f"debug: exp_z is {exp_z}")
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        print(f"debug: np.sum(exp_z, axis=1, keepdims=True) is {np.sum(exp_z, axis=1, keepdims=True)}")
        print(f"debug: np.sum(exp_z, keepdims=True) is {np.sum(exp_z, keepdims=True)}")
        return a

if __name__ == '__main__':
    z = np.array([[3,1,-3],[1,-3,3]]).reshape(2,3)
    a = Softmax().forward(z)
    print(a)
    
    