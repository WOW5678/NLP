# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/19 0019 下午 9:49
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from  tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import math_ops
import tensorflow as tf

def pointer_decoder(decoder_inputs,initial_state,attention_states,cell,
                    feed_prev=True,dtype=dtypes.float32,scope=None):
    '''
    RNN decoder with pointer net for sequence-sequence model.
    :param decoder_inputs:a list of 2D Tensors [batch_size x cell.input_size]
    :param initial_state:2D Tensor [batch_size x cell.state_size].
    :param attention_states:3D Tensor [batch_size x attn_length x attn_size].
    :param cell:rnn_cell.RNNCell defining the cell function and size.
    :param feed_prev:
    :param dtype:The dtype to use for the RNN initial state (default: tf.float32)
    :param scope:VariableScope for the created subgraph; default: "pointer_decoder".
    :return:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
        [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either i-th decoder_inputs.
        First, we run the cell
        on a combination of the input and previous attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      states: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
    '''
    if not decoder_inputs:
        raise ValueError('Must provide at least 1 input to attention decoder.')
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())

    with vs.variable_scope(scope or 'decoder_pointer'):
        batch_size=array_ops.shape(decoder_inputs[0])[0] # Needed for reshaping
        input_size=decoder_inputs[0].get_shape()[1].value
        # 有几个待选择项，attention_Length即为几，因为pre-net中attention的作用就是用来选择输入项的
        attn_length=attention_states.get_shape()[1].value
        attn_size=attention_states.get_shape()[2].value

        #  To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        # hidden:shape:[batch_size,maxlen+1,1,rnn_size]
        #[batch, in_height, in_width, in_channels]
        hidden=array_ops.reshape(attention_states,[-1,attn_length,1,attn_size])
        attention_vec_size=attn_size  # Size of query vectors for attention.
        #`[filter_height, filter_width, in_channels, out_channels]`
        k=vs.get_variable('AttnW',[1,1,attn_size,attention_vec_size])
        # hidden_features的shape不发生改变，还是[batch_size,maxlen+1,1,rnn_size]
        #相当于执行了W1Ej的操作

        hidden_features=nn_ops.conv2d(hidden,k,[1,1,1,1],'SAME')
        v=vs.get_variable('AttnV',[attention_vec_size])

        states=[initial_state]

        def attention(query):
            '''
            Point on hidden using hidden_features and query
            :param query:shape:[batch_size,attention_size]
            :return:
            '''
            with vs.variable_scope('Attention'):
                # y shape:[batch_size,attention_size]
                # 相当于执行W2Dj的运算
                y=core_rnn_cell._linear(query,attention_vec_size,True)
                # y shape:[batch_size,1,1,attention_size]
                y=array_ops.reshape(y,[-1,1,1,attention_vec_size])

                # Attention mask is softmax of v^T *tanh(...)
                s=math_ops.reduce_sum(v*math_ops.tanh(hidden_features+y),[2,3])
                return s

        outputs=[]
        prev=None
        # satck（)作用：行不变，增加列 shape:(2,)，其中第一个元素为？，第二个元素为attn_size
        batch_attn_size=array_ops.stack([batch_size,attn_size])
        # attns:[?,rnn_size]
        attns=array_ops.zeros(batch_attn_size,dtype=dtype)

        attns.set_shape([None,attn_size])
        inps=[]
        for i in range(len(decoder_inputs)):
            if i>0:
               # vs.get_variable_scope().resuse_variables()
                vs.get_variable_scope().reuse_variables()
            # inp:[batch_size,1]
            inp=decoder_inputs[i]

            if feed_prev and i>0:

                inp=tf.stack(decoder_inputs) #[11,30,1]
                inp=tf.transpose(inp,perm=[1,0,2]) #(30,11,1)
                inp=tf.reshape(inp,[-1,attn_length,input_size])
                inp=tf.reduce_sum(inp*tf.reshape(tf.nn.softmax(output),[-1,attn_length,1]),1) #(30,1)
                inp=tf.stop_gradient(inp)
                inps.append(inp)

            # Use the same inputs in inference, order internaly
            # Merge input and previous attentions into one vector of right size
            #？？？？？为什么要对inp进行线性变化呢？ 不是直接输入进去就可以了吗？
            x=core_rnn_cell._linear([inp,attns],input_size,True)

            # Run the RNN
            # with tf.variable_scope(name_or_scope='cell',reuse=tf.AUTO_REUSE):

            # print('x:',x) # (batch_size,rnn_size)
            # print('states[-1]:',states[-1]) #(batch_size,rnn_size)
            cell_output,new_state=cell(x,states[-1])
            states.append(new_state)
            # Run the attention mechnism
            output=attention(new_state)
            outputs.append(output)
    return outputs,states,inps
