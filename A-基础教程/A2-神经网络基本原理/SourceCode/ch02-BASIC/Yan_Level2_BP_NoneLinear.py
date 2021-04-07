# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:14:39 2021

@author: yan
"""

import numpy as np
import matplotlib.pyplot as plt

def draw_fun(X, Y):
    x = np.linspace(1.2, 10)
    a=x*x
    b=np.log(a)
    c=np.sqrt(b)
    
    plt.plot(x,c) #这里就是绘画出x&c的关系
    plt.plot(X,Y, 'bx')
    
    d = 1/(x*np.sqrt(np.log(x**2))) #对应的是数学解析解
    plt.plot(x,d) #这里就是绘画出数学解析解，x&d的关系
    
    """
    plt.show(), for now, I dont think it is good to use this command
    """
    
def forward(x):
    a = x*x
    b=np.log(a)
    c=np.sqrt(b)
    return a,b,c

def backward(x,a,b,c,y):
    loss=c-y
    delta_c=c-y
    delta_b=delta_c*2*np.sqrt(b)
    delta_a=delta_b*a
    delta_x=delta_a/2/x
    return loss,delta_c,delta_b,delta_a,delta_x    

def update(x,delta_x):
    x = x - delta_x
    return x

if __name__ == '__main__':
    print("How to play, (1) input x, (2) calculate c (3) input target number but not far from c")
    print("input x as initial value between 1.2 and 10, you can try 1.5")
    # line = input()
    # x = float(line)
    x = float(1.3)
    
    a,b,c=forward(x)
    print("c=%f" %c)
    print("Input y as target number (0.5, 2), you can try 1.9")
    """
    line = input()
    y = float(line)
    """
    y = float(1.9)
    accuracy = 1e-5
    
    X,Y =[],[]
    
    for i in range(20):
        print("forward...")
        a,b,c=forward(x)
        print("x=%f, a=%f, b=%f, c=%f"%(x,a,b,c))
        X.append(x)
        Y.append(c)
        
        print("backward...")
        loss,delta_c,delta_b,delta_a,delta_x=backward(x,a,b,c,y)
        if(abs(loss)<accuracy):
            print("done!")
            break
        
        print("update...")        
        x=update(x,delta_x)
        print("delta_c=%f, delta_b=%f, delta_c=%f, delta_x=%f"%(delta_c, delta_b, delta_a, delta_x))
        print("x=%f"%x)
        
    draw_fun(X,Y)