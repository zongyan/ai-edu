# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:02:27 2021

@author: yan
"""

import numpy as np

def target_function(w, b):
    x=2*w+3*b
    y=2*b+1
    z=x*y
    return x,y,z

def double_variable(w, b, t):
    print("double variable function: w, b ------")
    accuracy = 1e-5
    while(True):
        x, y, z = target_function(w, b)
        delta_z = z - t
        print("w=%f, b=%f, z=%f, delta_z=%f"%(w, b, z, delta_z))
        if(abs(delta_z)<accuracy):
            break
        
        delta_b = delta_z/(3*y+2*x)/2
        delta_w = delta_z/(2*y)/2
        print("delta_b=%f, delta_w=%f"%(delta_b, delta_w))
        b=b-delta_b
        w=w-delta_w
        
    print("done!")
    print("final b is %f"%(b))
    print(f"final w is {w}")    
    
if __name__ == '__main__':
    w = 3
    b = 4
    t = 150
    double_variable(w,b,t)
        