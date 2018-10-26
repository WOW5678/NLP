# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/19 0019 下午 8:34
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 实现一个简单的pointer network网络完成对数字的排序
 通过观察发现，最后一个轮次的结果并不是很好，因此需要加入中间模型保存的代码，根据识别准确率保存最佳的模型
 这一块还没加入
 tf version:1.9.0

"""
import tensorflow as tf
import numpy as np
import os
from dataset import DataGenerator
from pointer import pointer_decoder


flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_integer('batch_size',128,'Batch_size')
flags.DEFINE_integer('max_steps',4,'Number of numbers to sort')
flags.DEFINE_integer('rnn_size',32,'RNN_SIZE')

class PointerNetwork(object):
    def __init__(self,max_len,input_size,size,num_layers,max_gradient_norm,batch_size,learning_rate,learning_reate_decay_factor):
        '''
        Create a simple network that handles only sorting
        :param max_len: maximum length of model
        :param input_size:size of input data
        :param size:number of units in each layer in the model
        :param num_layers:number of layers in the model
        :param max_gradient_norm:gradients will be clipped to maximally this norm.
        :param batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
        :param learning_rate:learning rate to start with.
        :param learning_reate_decay_factor:decay learning rate by this much when needed.
        '''
        self.batch_size=batch_size
        self.learning_rate=tf.Variable(float(learning_rate),trainable=True)
        self.learning_rate_decay_op=self.learning_rate.assign(
            self.learning_rate*learning_reate_decay_factor
        )
        self.global_step=tf.Variable(0,trainable=False)
        # with tf.variable_scope(name_or_scope='cell',reuse=tf.AUTO_REUSE):
        # if num_layers>1:

        with tf.variable_scope('cell',reuse=tf.AUTO_REUSE):
            #cell=tf.contrib.rnn.GRUCell(size)
            cell=tf.nn.rnn_cell.BasicRNNCell(size)
            #注意：这里一旦num_layers不为1 就报维度错误，具体原因不清楚
            if num_layers>1:
                cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])
        # init_state = cell.zero_state(batch_size, dtype=tf.float32)
        #cell=tf.nn.rnn_cell.MultiRNNCell([self.get_cell(size)],state_is_tuple=True)
        # else:
        #     cell=self.get_cell(size)
        self.encoder_inputs=[]
        self.decoder_inputs=[]
        self.decoder_targets=[]
        # weights后面传入的值其实全为1，并没有太大的意义
        self.target_weights=[]
        # 这样写的目的是为了可以输入到RNN模型中，而不用再次执行split操作
        for i in range(max_len):
            self.encoder_inputs.append(tf.placeholder(
                tf.float32,[batch_size,input_size],name='EncoderInput%d'%i
            ))
        for i in range(max_len+1):
            self.decoder_inputs.append(tf.placeholder(
                tf.float32,[batch_size,input_size],name='DecoderInput%i'%i))
            self.decoder_targets.append(tf.placeholder(
                tf.float32,[batch_size,max_len+1],name='DecoderTarget%d'%i)) # one hot
            self.target_weights.append(tf.placeholder(
                tf.float32,[batch_size,1],name='TargetWeight%d'%i))

            # Encoder

        # Need for attention
        encoder_outputs,final_state=tf.contrib.rnn.static_rnn(cell,self.encoder_inputs,dtype=tf.float32)
        # Need a dummy ouput to point on it. End of decoding
        # 相当与在0位置添加了一种选择，当选择到了0位置以后，即可结束decoder部分
        encoder_outputs=[tf.zeros((FLAGS.batch_size,FLAGS.rnn_size))]+encoder_outputs
        print('encoder_ouputs:',encoder_outputs)
        # First calculate a concatenation of encoder outputs to put attention on
        top_state=[tf.reshape(e,[-1,1,cell.output_size]) for e in encoder_outputs]
        # attention_states:shape:(batch_size,max_len+1,rnn_size)
        attention_states=tf.concat(values=top_state,axis=1)

        with tf.variable_scope('decoder'):
            outputs,states,_=pointer_decoder(
                self.decoder_inputs,final_state,attention_states,cell)

        # 这个代码测试写的比较简单，用的就是训练集做为测试

        with tf.variable_scope('decoder',reuse=True):
            predictions, _, inps = pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell, feed_prev=True)
        # predictions:list,每个元素为（30，11）
        #final state:(30,32)
        self.predictions=predictions
        self.outputs=outputs
        self.inps=inps
    def get_cell(self,size):
        return tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

    def create_feed_dict(self,encoder_input_data, decoder_input_data, decoder_target_data):
        feed_dict={}
        for placeholder,data in zip(self.encoder_inputs,encoder_input_data):
            feed_dict[placeholder]=data
        for placeholder,data in zip(self.decoder_inputs,decoder_input_data):
            feed_dict[placeholder]=data
        for placeholder ,data in zip(self.decoder_targets,decoder_target_data):
            feed_dict[placeholder]=data

        for placeholder in self.target_weights:
            feed_dict[placeholder]=np.ones([self.batch_size,1])
        return feed_dict

    def step(self):
        loss=0.0
        for output,target,weights in zip(self.outputs, self.decoder_targets, self.target_weights):
            loss+=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=target)*weights
        loss=tf.reduce_mean(loss)

        test_loss=0.0
        for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
            test_loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight
        test_loss = tf.reduce_mean(test_loss)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        train_loss_value = 0.0
        test_loss_value = 0.0
        correct_order = 0
        all_order = 0
        num_epochs=1000

        with tf.Session() as sess:
            # merged=tf.merge_all_summaries()
            # writer=tf.train.SummaryWriter('/tmp/pointer_logs',sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(num_epochs):
                # 每一轮过去都要重新统计一下准确率
                correct_order = 0
                all_order = 0

                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps)

                # Train
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                d_x, l = sess.run([loss, train_op], feed_dict=feed_dict)
                # 为什么要乘以一个系数呢？？？
                train_loss_value = 0.9 * train_loss_value + 0.1 * d_x

                # 每 50轮打印一下训练集的loss,并且计算一下测试集的损失值
                if i % 50 == 0:
                    print('Step: %d' % i)
                    print("Train: ", train_loss_value)

                    encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                        FLAGS.batch_size, FLAGS.max_steps, train_mode=False)
                    # Test
                    feed_dict = self.create_feed_dict(
                        encoder_input_data, decoder_input_data, targets_data)
                    inps_ = sess.run(self.inps, feed_dict=feed_dict)

                    predictions = sess.run(self.predictions, feed_dict=feed_dict)

                    test_loss_value = 0.9 * test_loss_value + 0.1 * sess.run(test_loss, feed_dict=feed_dict)
                    print("Test: ", test_loss_value)

                #每100轮之后，统计一下预测的准确值
                if i % 100 == 0 or i==num_epochs-1:
                    print("Test: ", test_loss_value)
                    tmp=[np.expand_dims(prediction, 0) for prediction in predictions]
                    tmp2=np.concatenate(tmp)
                    predictions_order = np.concatenate([np.expand_dims(prediction, 0) for prediction in predictions],axis=0)
                    predictions_order = np.argmax(predictions_order, 2).transpose(1, 0)[:, 0:FLAGS.max_steps]

                    input_order = np.concatenate(
                        [np.expand_dims(encoder_input_data_, 0) for encoder_input_data_ in encoder_input_data])
                    # squeeze()函数会将维度为1的维度进行压缩
                    # +1是要干嘛？？？因为target预测时是5个数，所以他的索引值最大为4 要整天比input_order的值大1，因为要对
                    #input_order的索引值加上1
                    input_order = np.argsort(input_order, 0).squeeze().transpose(1, 0) + 1


                    print('input_order:',input_order)
                    print('prediction_order:',predictions_order)
                    # 只有全部排列正确才会被统计进去
                    correct_order += np.sum(np.all(predictions_order == input_order,
                                                   axis=1))
                    all_order += FLAGS.batch_size



                    print('Correct order / All order: %f' % (correct_order / all_order))


                    # print(encoder_input_data, decoder_input_data, targets_data)
                    # print(inps_)

print(tf.__version__)
pointer_network=PointerNetwork(FLAGS.max_steps,1,FLAGS.rnn_size,1,5,FLAGS.batch_size,1e-2,0.95)
dataset=DataGenerator()
pointer_network.step()


