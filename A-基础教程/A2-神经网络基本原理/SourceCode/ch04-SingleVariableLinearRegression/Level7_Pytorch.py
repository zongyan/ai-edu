import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from torch.utils.data import TensorDataset, DataLoader
from HelperClass.DataReader_1_0 import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

file_name = "../../SourceCode/Data/ch04.npz"


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    max_epoch = 500
    num_category = 3
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()

    num_input = 1       # input size
    # get numpy form data
    XTrain, YTrain = sdr.XTrain, sdr.YTrain
    print(f"\ndebug: XTrain is {XTrain} \n")
    print(f"\ndebug: YTrain is {YTrain} \n")
    
    torch_dataset = TensorDataset(torch.FloatTensor(XTrain), torch.FloatTensor(YTrain))
 
    print(f"\ndebug: torch.FloatTensor(XTrain) is {torch.FloatTensor(XTrain)} \n")
    print(f"\ndebug: torch.FloatTensor(YTrain) is {torch.FloatTensor(YTrain)} \n")
    print(f"\ndebug: torch_dataset is {torch_dataset} \n")


    train_loader = DataLoader(          # data loader class
        dataset=torch_dataset,
        batch_size=32,
        shuffle=True,
    )

    loss_func = nn.MSELoss()
    model = Model(num_input)
    # model = Model().to("cpu") # specify the CPU for use
    optimizer = Adam(model.parameters(), lr=1e-2)

    model.train() # set mode
    e_loss = []     # mean loss at every epoch
    for epoch in range(max_epoch):
        b_loss = []     # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred,batch_y)
            b_loss.append(loss.cpu().data.numpy()) # change from tensor to array
            """
            .cpu() copies the tensor to the CPU.
            .numpy() creates a NumPy array from the tensor. The tensor and the 
            array share the underlying memory, therefore if the NumPy array is 
            modified in-place, the changes will be reflected in the original tensor.
            """
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



