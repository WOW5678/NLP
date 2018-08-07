# -*- coding:utf-8 -*-
'''
build 多变量时间序列预测问题的LSTM
预测北京下一个小时的污染程度
'''
import csv
from pandas import read_csv
from datetime import datetime
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
from keras.models import  Sequential
from keras.layers.recurrent import  LSTM
from keras.layers.core import Dense

# load data
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')
dataset=read_csv('raw.csv',parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)
#delete the column of 'No'
dataset.drop('No',axis=1,inplace=True)
print('dataset:',dataset)
# manually specify column names
# 重命名
dataset.columns=['polluation','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.name='date'

#mark all NA values with 0
dataset['polluation'].fillna(0,inplace=True)

#drop the first 24 hours 因为前24个小时的数据是空的
dataset=dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
#save to file
dataset.to_csv('polluation.csv')

from matplotlib import pyplot
# load the dataset
dataset=read_csv('polluation.csv',header=0,index_col=0)
values=dataset.values
#specify columns to plot
groups=[0,1,2,3,5,6,7]
i=1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups),1,i)
    #画出当前列的分布图
    pyplot.plot(values[:,group])
    pyplot.title(dataset.columns[group],y=0.5,loc='right')
    i+=1
pyplot.show()

# Covert series to supervised learning
def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=DataFrame(data)
    cols,names=list(),list()
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)')%(j+1,i) for j in range(n_vars)]
    # forecast sequence (t,t+1,t+2...)
    for i in range(0,n_out):
        cols.append(df.shift(-1))
        if i==0:
            names+=[('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]

    #put it all together
    agg=pd.concat(cols,axis=1)
    agg.columns=names


    #drop rows with NAn Values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#load the dataset
dataset=read_csv('polluation.csv',header=0,index_col=0)
values=dataset.values
# integer encode direction
encoder=LabelEncoder()
#需要对风向这一列的数据进行one-hot编码
values[:,4]=encoder.fit_transform(values[:,4])

#ensure all data is features
#把所有的数据规整到一个指定的范围内
scaler=MinMaxScaler(feature_range=(0,1))
scalerd=scaler.fit_transform(values)
print('scalerd:',scalerd)
# frame as supervised learning
reframed=series_to_supervised(scalerd,1,1)
print('reframed:',reframed)
# drop columns we don't want to predict

#其他几列的预测值我们并不关注 而只关注对polluation列的观察值 所以删除其他的列
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]],axis=1,inplace=True)
print('after delete:',reframed.head(5))


# Define and matching data
#split data into train and test
values=reframed.values
n_train_hours=365*24
train=values[:n_train_hours,:]
test=values[n_train_hours:,:]
# split into input and outputs
train_x,train_y=train[:,:-1],train[:,-1]
test_x,test_y=test[:,:-1],test[:,-1]

#reshape input to be 3D[samples,timesteps, features]
train_x=train_x.reshape((train_x.shape[0],1,train_x.shape[1]))
test_x=test_x.reshape((test_x.shape[0],1,test_x.shape[1]))
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

#design network
model=Sequential()
#LSTM中input_shape的格式为（样本数，时间步长，特征数）
model.add(LSTM(50,input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dense(1))
#均方差作为损失值 adam作为优化算法
model.compile(loss='mae',optimizer='adam',metrics=['accuracy'])

#fit network
history=model.fit(train_x,train_y,epochs=50,batch_size=72,validation_data=(test_x,test_y))
print('history.history:',history.history)
#plot history
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.show()

