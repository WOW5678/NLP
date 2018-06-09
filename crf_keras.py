# -*- coding:utf-8 -*-
'''

'''
from keras.layers import Layer
import keras.backend as k

class CRF(Layer):
    '''
    纯kears实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只能用来训练模型，而预测需要另外建立模型
    '''
    def __init__(self,ignore_last_label=False,**kwargs):
        print('init function is called.')
        ''''
        ignore_last_label:定义要不要忽略最后一个标签，起到mask的效果
        '''
        self.ignore_last_label=1 if ignore_last_label else 0
        super(CRF,self).__init__(**kwargs)

    # 这是定义权重的方法，可训练的权应该在这里被加入列表`self.trainable_weights中
    def build(self, input_shape):
        print('build function is called.')
        print('input_shape:',input_shape)
        self.num_labels=input_shape[-1]-self.ignore_last_label
        self.trans=self.add_weight(name='crf_trans',shape=(self.num_labels,self.num_labels),
                                   initializer='glorot_uniform',
                                   trainable=True)
    def log_norm_step(self,inputs,states):
        '''
        递归计算归一化因子
        要点：1. 递归计算 2.用logsumexp避免溢出
        技巧：通过expand_dims来对齐张量
        :param inputs: shape：(batch_size,None-1,output_dim)
        :param states: shape(batch_size,output_dim)
        :return:
        '''

        print('log_norm_step function is called.')
        #expand_dims(inputs,axis)#在指定维度上增加维度
        states=k.expand_dims(states[0],2)  #(batch_size,output_dim,1)
        trans=k.expand_dims(self.trans,0)  #(1,output_dim,output_dim)
        output=k.logsumexp(states+trans,1) # (batch_size,ouput_dim)
        print('outputs+inputs:',(output+inputs).shape) # (batch_size,ouput_dim)
        return output+inputs,[output+inputs]

    def path_score(self,inputs,labels):
        '''
        计算目标路径的相对概率（还没有进行归一化）
        要点：逐标签得分，加上转移概率得分
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分
        :param inputs: 预测值
        :param labels: 目标值
        :return:目标路径的得分
        '''
        print('path_score function is called.')
        point_score=k.sum(k.sum(inputs*labels,2),1,keepdims=True) # 逐标签得分
        #在下标为dim的轴上增加一维
        labels1=k.expand_dims(labels[:,:-1],3)
        print('labels1:',labels1.shape)
        labels2=k.expand_dims(labels[:,1:],2)
        print('labels2:',labels2.shape)
        labels=labels1*labels2 # 两个错位labels 负责从转移矩阵中抽取目标转移得分
        trans=k.expand_dims(k.expand_dims(self.trans,0),0)
        tran_score=k.sum(k.sum(trans*labels,[2,3]),1,keepdims=True)
        return point_score+tran_score

    #这是定义层功能的方法
    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        print('call function is called.')
        return inputs
    # y_pred需要是one hot形式
    def loss(self,y_true,y_pred):
        print('loss function is called.')
        #mask什么作用？？？？
        mask=1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred=y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_state=[y_pred[:,0]]

        #k.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None,
        #  constants=None, unroll=False, input_length=None)
        #在张量的时间维上迭代
        # self.log_norm_step:每个时间步要执行的函数其参数
        #input：形如(samples, ...)的张量，不含时间维，代表某个时间步时一个batch的样本
        #返回值：形如(last_output, outputs, new_states)的tuple

        # 计算Z向量（对数）
        log_norm,_,__=k.rnn(self.log_norm_step,y_pred[:,1:],init_state,mask=mask)
        log_norm = k.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        path_score=self.path_score(y_pred,y_true) #计算分子（对数）
        return log_norm-path_score #即log(分子/分母)
    ## 训练过程中显示逐帧准确率的函数，排除了mask的影响
    def accuracy(self,y_true,y_pred):
        print('accuracy function is called.')
        mask=1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred=y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal=k.equal(k.argmax(y_true,2),k.argmax(y_pred,2))
        isequal=k.cast(isequal,'float32')
        if mask==None:
            return k.mean(isequal)
        else:
            return k.sum(isequal*mask)/k.sum(mask)

