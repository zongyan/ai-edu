import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from torch.utils.data import TensorDataset, DataLoader
from HelperClass.NeuralNet_1_2 import *
from HelperClass.HyperParameters_1_1 import *
from HelperClass.Visualizer_1_0 import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

file_name = "../../SourceCode/Data/ch06.npz"

def ShowRawData(x, y):    
    fig = plt.figure(figsize=(6.5,6.5)) # width=6.5inches, height=6.5inches
    for i in range(200):
        if y[i] == 1:
            plt.scatter(x[i,0], x[i,1], marker='x', c='g')
        else:
            plt.scatter(x[i,0], x[i,1], marker='o', c='r')
    plt.show()    
    
def ShowResult(trained_weight, trained_bias, x, y):
    print(f"debug: trained_weight is {trained_weight} ")
    # [[-0.29215586  0.31718323], [-0.19612052 -0.23099351]]
    print(f"debug: trained_bias is {trained_bias}")    
    # [0.2441295 -0.4503359] 

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 2)
    def forward(self, x):
        x = self.fc(x) # 为什么在linear multiple classification就是会使用softmax
                       # function, 但是在这个linear binary classification里面，
                       # 就是没有使用logistic function（但是在另外一个版本的代码里
                       # 面，就是使用的了），参见line 90
        return x

if __name__ == '__main__':
    max_epoch = 100
    num_category = 3
    sdr = DataReader_1_1(file_name)
    sdr.ReadData()
    data_num = np.shape(sdr.YTrain)[0]
    print(f"\ndata_num is : {data_num} \n")        
    num_input = 2       # input size
    # get numpy form data
    XTrain, YTrain = sdr.XTrain, np.reshape(sdr.YTrain, [data_num, ])
    # print(f"\nXTrain is : {XTrain} \n")
    # print(f"\nYTrain is : {YTrain} \n")    
    ShowRawData(XTrain, YTrain)
    torch_dataset = TensorDataset(torch.FloatTensor(XTrain), torch.LongTensor(YTrain))

    train_loader = DataLoader(          # data loader class
        dataset=torch_dataset,
        batch_size=32,
        shuffle=True,
    )

    loss_func = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    model = Model(num_input)
    optimizer = Adam(model.parameters(), lr=1e-4)

    e_loss = []     # mean loss at every epoch
    for epoch in range(max_epoch):
        b_loss = []     # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred,batch_y)
            b_loss.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            b_loss.append(loss.cpu().data.numpy())
        e_loss.append(np.mean(b_loss))
        if epoch % 20 == 0:
            print("Epoch: %d, Loss: %.5f" % (epoch, np.mean(b_loss)))
    plt.plot([i for i in range(max_epoch)], e_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.show()
    
    ShowResult(model.fc.weight.data.numpy(), model.fc.bias.data.numpy(),
               XTrain, YTrain)        
    # 后来就是把weight & bias打印出来之后，就是发现这个输出的dimension不是1，而是2；
    # 这个就是和另外一个版本的linear binary classification的不一样，根据tutorial&
    # 代码来看，output dimension应该是1。所以这个代码，不是linear binary classification
    # 的代码了。如果是需要看linear classification的代码，请参见linear multiple 
    # classification


