# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import pandas as pd
from pandas import read_csv

"""
其实这个代码就是一个简易的处理数据的模板了，以后可以借鉴使用这段代码了。感觉一般情况下，
就是使用pandas或者是numpy导入数据，然后对数据做相应的处理。如果是使用pandas导入数据的
话，最后还是需要转换成numpy的array格式了。
"""

def set_pollution_value(ds, start, end, start_value, end_value):
    step_value = (end_value - start_value) / (end - start)
    value = start_value + step_value
    for i in range(start+1, end, 1):
        ds.iat[i,3] = value
        value = value + step_value
    return ds

def get_previous_pollution_value(ds, row, col):
    prev_value = 0
    if (row - 1 >= 0):
        prev_value = ds.iat[row-1,col]
    return prev_value

def get_next_pollution_value(ds, row, col):
    next_value = 0
    if (row < len(ds)):
        next_value = ds.iat[row,col]
    return next_value


#dataset = read_csv('../../data/PM25_data.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset = read_csv('../../SourceCode/data/PM25_data.csv') # 这个用的是pandas来导入数据的, 感觉要么是pandas，要么是numpy导入数据
dataset.drop('No', axis=1, inplace=True)  # 这个用的是舍弃某一列的数据
dataset.drop('year', axis=1, inplace=True)

# manually specify column names， 重命名每一列的数据
dataset.columns = ['month', 'day', 'hour', 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'rain', 'snow']
#dataset.index.name = 'date'
dataset = dataset.replace('NW',9)
dataset = dataset.replace('NE',3)
dataset = dataset.replace('SE',6)
dataset = dataset.replace('SW',12)
ds = dataset.replace('cv',0)

total = len(ds)
# fill NaN pollution value， 这段代码还是设计巧妙的，以后也学可以借鉴，就是弥补loss
# data的方式，当然，这个方式也是在文本中由了相应的介绍。
for i in range(total):
    if (pd.isnull(ds.iat[i,3])):
        start = i - 1
        # read previous pollution value
        prev_value = get_previous_pollution_value(ds, i, 3)
        for j in range(i+1, total, 1):
            if (pd.isnull(ds.iat[j,3]) == False):
                end = j 
                break
        next_value = get_next_pollution_value(ds, j, 3)
        print(start,end)
        ds = set_pollution_value(ds, start, end, prev_value, next_value)


# process accumulated wind_speed value， 这段代码，是处理累计数的一段代码了。
prev_value = 0
prev_dir = 0
for i in range(total):
    current_dir = ds.iat[i,7]
    accumulated_value = ds.iat[i,8]
    if (current_dir == prev_dir):
        current_value = accumulated_value - prev_value
        ds.iat[i,8] = current_value
    #endif
    prev_value = accumulated_value
    prev_dir = current_dir

print(ds.head(24)) # 打印出前面24行的数据

# shift y value up
# reason: the current situation continous for 1 hour to make next hour's PM2.5 value
# 当前的气象条件（及当前污染指数值）持续一小时后会产生下一个小时的污染指数
pollution_y = np.zeros((total,1))
tmp = ds['pollution'].to_numpy()
count = tmp.shape[0]
pollution_y[0:count-1] = tmp[1:count].reshape(-1,1)
pollution_y[-1] = tmp[-1]*2 - tmp[-2]

# ds.drop('month', axis=1, inplace=True)
# ds.drop('day', axis=1, inplace=True)
# ds.drop('hour', axis=1, inplace=True)
# ds.drop('rain', axis=1, inplace=True)
# ds.drop('snow', axis=1, inplace=True)

print(ds.head(24))

pollution_x = ds.to_numpy() # Convert the DataFrame to a NumPy array

np.savez("../../SourceCode/data/ch19_pm25_train.npz", data=pollution_x[0:total-8760], label=pollution_y[0:total-8760])
np.savez("../../SourceCode/data/ch19_pm25_test.npz", data=pollution_x[total-8760:], label=pollution_y[total-8760:])
