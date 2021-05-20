from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam # 是从torch.optim导入Adam
import torch.nn.init as init
import warnings
warnings.filterwarnings('ignore')

train_file = "../../SourceCode/Data/ch14.Income.train.npz"
test_file = "../../SourceCode/Data/ch14.Income.train.npz"

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(14, 32, bias=True)
        self.fc2 = nn.Linear(32, 16, bias=True)
        self.fc3 = nn.Linear(16, 8, bias=True)
        self.fc4 = nn.Linear(8, 4, bias=True)
        self.fc5 = nn.Linear(4, 2, bias=True)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear): # 判断m是不是全链接层
                init.xavier_uniform_(m.weight, gain=1)
                print(m.weight)

if __name__ == '__main__':
    # reading data
    dataReader = LoadData()

    max_epoch = 500     # max_epoch
    batch_size = 64         # batch size
    lr = 1e-4               # learning rate

    # define model
    model = Model()
    model._initialize_weights()     # init weight

    # loss and optimizer
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    num_train = dataReader.YTrain.shape[0]
    num_val = dataReader.YDev.shape[0]

    """
    其实之前网上找的regression tutorial和这个tutorial在数据处理上还是有相似之初。
    比如说，都是先是处理完数据，然后就是把数据转成tensor的格式（如使用torch.tensor或
    torch.FloatTensor或torch.LongTensor的形式）
    
    然后接着就是用TensorDataset把数据wrapping起来（不过这个步骤，在regression 
    tutorial是没有看见的）。我不认为是两者谁有错误，而是说regression tutorial的方式
    就是高级一些，有部分功能可能就是隐藏性的做了，比如定义class HouseDataset(T.utils.data.Dataset):
    的时候。_____所以，暂时还是先使用本tutorial的方式，等时机成熟了，再转。
    
    最后，就是可以使用DataLoader对数据进行载入了。    
    """
    torch_dataset = TensorDataset(torch.FloatTensor(dataReader.XTrain), torch.LongTensor(dataReader.YTrain.reshape(num_train,)))
    XVal, YVal = torch.FloatTensor(dataReader.XDev), torch.LongTensor(dataReader.YDev.reshape(num_val,))
    train_loader = DataLoader(  # data loader class
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    et_acc = []        # store training loss
    ev_acc = []        # store validate loss

    for epoch in range(max_epoch):
        bt_acc = []  # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()         # backward
            optimizer.step()
            prediction = np.argmax(pred.cpu().data, axis=1)
            bt_acc.append(accuracy_score(batch_y.cpu().data, prediction))
        val_pred = np.argmax(model(XVal).cpu().data,axis=1)
        bv_acc = accuracy_score(dataReader.YDev,val_pred)
        et_acc.append(np.mean(bt_acc))
        ev_acc.append(bv_acc)
        print("Epoch: [%d / %d], Training Acc: %.6f, Val Acc: %.6f" % (epoch, max_epoch, np.mean(bt_acc), bv_acc))


    plt.plot([i for i in range(max_epoch)], et_acc)        # training loss
    plt.plot([i for i in range(max_epoch)], ev_acc)        # validate loss
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.show()

    """
    参见下面的链接，就是可以知道accuracy_score的作用了：
    https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

    不过我个人感觉，这个accuracy_score主要是用在classification里面了。
    """






