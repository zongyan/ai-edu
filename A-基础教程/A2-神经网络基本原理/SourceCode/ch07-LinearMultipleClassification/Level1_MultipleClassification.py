# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_2 import *

file_name = "../../SourceCode/Data/ch07.npz"

def inference(net, reader):
    xt_raw = np.array([5,1,7,6,5,6,2,7]).reshape(4,2)
    xt = reader.NormalizePredicateData(xt_raw)
    output = net.inference(xt)
    r = np.argmax(output, axis=1)+1
    print(f"debug: output is {output}")
    print(f"debug: np.argmax(output) is {np.argmax(output)}")
    print(f"debug: np.argmax(output, axis=0) is {np.argmax(output, axis=0)}")
    # 使用axis=0值表示沿着每一列向下执行方法
    print(f"debug: np.argmax(output, axis=1) is {np.argmax(output, axis=1)}")    
    # 使用axis=1值表示沿着每一行向执行对应的方法
    print("output=", output)
    print("r=", r)

# 主程序
if __name__ == '__main__':
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    num_input = 2
    """
    之前就是一直是纠结这个parameters的调用怎么是可以使用class的方式的了。这会让总算
    是明白的了。首先就是使用HyperParameters_1_1定义了一个类，这个类中就是会存在一些的
    Hyper Parameters； 
    然后就是可以在这里就是示例化，如下一行的代码所示。然后接着就是可以把这个实例化的
    变量（本代码中是params）传递给NeuralNet_1_2。
    又是因为在NeuralNet_1_2中的时候，在使用def的时候，也是使用了class的概念（如
    NeuralNet_1_2的line 21所示），所以就是可以在这个def里面，就是使用
    HyperParameters_1_1定义的一些变量了。另外，这里就是需要注意的是，就是在NeuralNet_1_2
    里面，需要import HyperParameters_1_1类似的代码了（如line 15所示）
    """
    params = HyperParameters_1_1(num_input, num_category, eta=0.1, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)

    inference(net, reader)
