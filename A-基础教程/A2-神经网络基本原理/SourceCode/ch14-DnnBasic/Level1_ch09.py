# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *

train_file = "../../SourceCode/Data/ch09.train.npz"
test_file = "../../SourceCode/Data/ch09.test.npz"

def ShowResult(net, dr):
    fig = plt.figure(figsize=(12,5))

    axes = plt.subplot(1,2,1)
    axes.plot(dr.XTest[:,0], dr.YTest[:,0], '.', c='g')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    axes.plot(TX, TY, 'x', c='r')
    axes.set_title("fitting result")

    axes = plt.subplot(1,2,2)
    y_test_real = net.inference(dr.XTest)
    axes.scatter(y_test_real, y_test_real-dr.YTestRaw, marker='o', label='test data')
    axes.set_title("difference")
    plt.show()

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    #dr.NormalizeX()
    #dr.NormalizeY(YNormalizationMethod.Regression)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

def model():
    dataReader = LoadData()
    num_input = 1
    num_hidden1 = 4
    num_output = 1

    max_epoch = 10000
    batch_size = 10
    learning_rate = 0.5

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    net = NeuralNet_4_0(params, "Level1_CurveFittingNet")
    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    fc2 = FcLayer_1_0(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")

    """
    上面这一段代码中，之所以使用这个add_layer的形式，就是为了避免大量的重复性的代码，
    作者就是使用了add_layer的function来完成这个步骤的，具体点说，是用了layer_list和
    layer_name两个list变量，然后就是可以把layer的层数和layer的名字对应上的了，然后
    每次添加一个layer，就是使用append即可。
    所以，每次就是先是初始化每一层的input，output，weights&bias等参数，然后就是使用
    add_layer这个函数就是可以把每一层连接成为一个neural network。
    
    另外，就是在这个training的时候，比如__forward， __backward， __update等函数就是
    使用的是for loop的形式，逐层进行计算，得到output，gradient，更新的weight&bias数值
    """
    
    net.train(dataReader, checkpoint=100, need_test=True)

    net.ShowLossHistory()
    ShowResult(net, dataReader)

if __name__ == '__main__':
    model()
