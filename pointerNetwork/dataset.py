# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/19 0019 下午 8:36
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 生成一些简单的数据
"""
import  numpy as np

class DataGenerator(object):
    def __init__(self):
        pass
    def next_batch(self,batch_size,N,train_mode=True):
        '''
        return the next "batch_size" examples from this data set
        :param batch_size:
        :param N: sentence length??
        :param train_mode: bool
        :return:
        '''
        # A sequence of random numbers from [0,1]
        reader_input_batch=[]

        #Sorted sequence that we feed to encoder
        # In inference we feed an unordered sequence again
        decoder_input_batch=[]

        #Ordered sequence where on hot vector encodes position in the input array
        writer_outputs_batch=[]

        for _ in range(N):
            reader_input_batch.append(np.zeros([batch_size,1]))
        for _ in range(N+1):
            decoder_input_batch.append(np.zeros([batch_size,1]))
            writer_outputs_batch.append(np.zeros([batch_size,N+1]))

        for b in range(batch_size):
            # np.random.rand(N):产生一个包含N个元素，每个值都在0-1之间的数的行向量
            sequence=np.sort(np.random.rand(N))
            shuffle = np.random.permutation(N)
            #print('shuffle:',shuffle)
            shuffled_sequence=sequence[shuffle]

            for i in range(N):
                 reader_input_batch[i][b]=shuffled_sequence[i]
                 if train_mode:
                     decoder_input_batch[i+1][b]=sequence[i]
                 else:
                     decoder_input_batch[i+1][b]=shuffled_sequence[i]

                 # writer_ouputs_batch???是什么作用啊？？没看懂
                 writer_outputs_batch[shuffle[i]][b,i+1]=1.0
                 # print('writer_outputs_batch:',writer_outputs_batch)
            # pointers to the stop symbol
            writer_outputs_batch[N][b,0]=1.0
        # print('reader_input_batch:',reader_input_batch)
        # print('decoder_input_batch:',decoder_input_batch)
        # print('writer_outputs_batch:',writer_outputs_batch)
        return reader_input_batch,decoder_input_batch,writer_outputs_batch



if __name__ == '__main__':
    dataset=DataGenerator()
    dataset.next_batch(2,3)