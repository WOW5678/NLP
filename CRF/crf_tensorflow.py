# -*- coding:utf-8 -*-
'''
An example of API, which learns a CRF for some random data.
The linear layer in the examples can be replaced by any nerual networks
'''
import numpy as np
import tensorflow as tf

# Data setting
num_examples=10
num_words=20
num_features=100
num_tags=5

#Random features
x=np.random.rand(num_examples,num_words,num_features).astype(np.float32)
# Random tag indices representing the gold sequence
y=np.random.randint(num_tags,size=[num_examples,num_words]).astype(np.int32)
print(y)

# All the squences in this example have the same length,but they can be variable in a real model
sequence_length=np.full(num_examples,num_words-1,dtype=np.int32)
# sequence_lengh=[19 19 19 19 19 19 19 19 19 19]
print('sequence_lengths:',sequence_length)

# Train and evaluate the model
with tf.Graph().as_default():
    with tf.Session() as session:
        # Add the data the the tensorflow graph
        x_t=tf.constant(x)
        y_t=tf.constant(y)
        sequence_lengths_t=tf.constant(sequence_length)

        # compute unary scores from a linear layer
        weights=tf.get_variable('weights',[num_features,num_tags])
        matricized_x_t=tf.reshape(x_t,[-1,num_features])
        matricized_unary_scores=tf.matmul(matricized_x_t,weights)
        unary_scores=tf.reshape(matricized_unary_scores,[num_examples,num_words,num_tags])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time
        log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(
            unary_scores,y_t,sequence_lengths_t
        )
        # Compute the verterbi sequences and scores
        viterbi_sequence,viterbi_score=tf.contrib.crf.crf_decode(
            unary_scores,transition_params,sequence_lengths_t
        )
        # add a train op to turn the parameters
        loss=tf.reduce_mean(-log_likelihood)
        train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        session.run(tf.global_variables_initializer())
        mask=(np.expand_dims(np.arange(num_words),axis=0)
              < np.expand_dims(sequence_length,axis=1))
        total_labels=np.sum(sequence_length)

        #Train a fixed number of iteration
        for i in range(200):
            tf_viterbi_sequence,_=session.run([viterbi_sequence,train_op])
            if i%100==0 or i==100:
                correct_labels=np.sum((y==tf_viterbi_sequence)*mask)
                accuracy=100.0*correct_labels/float(total_labels)
                print('Accuracy:%.2f%%'%accuracy)
                print('tf_viterbi_sequence:',tf_viterbi_sequence)
                print('y:',y)
