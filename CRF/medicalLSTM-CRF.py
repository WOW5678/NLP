# -*- coding:utf-8 -*-
'''
function: LSTM-CRF实现对医疗数据的处理
1. 将CCKS数据处理成有标注的序列数据
2. 对每个word embedding-->LSTM-->CRF
3. 其中embedding过程： 每个病历文件为一个样本数据，每个单词变成一个id,每个病历文件
变成id序列，然后使用tf.nn.embedding_lookup()生成每个word embedding,然后送入LSTM-->CRF中
'''
import tensorflow as tf
import os
import glob
import codecs
import numpy as np
print(tf.__version__)

# 使用SMBE方法进行数据标注
def labelChar(all_txtoriginal_texts):
    fw=open('labelData.txt', 'w',encoding='utf-8')
    label_dict={'解剖部位':'body','手术':'surgery','药物':'drug',
                '独立症状':'ind_symptoms','症状描述':'SymptomDes'}
    allSamples=[]
    allSample_labels=[]
    for file in all_txtoriginal_texts:
        original_filename=file
        label_filename=file.replace('txtoriginal.','')
        with codecs.open(original_filename,encoding='utf-8') as f:
            original_content=f.read().strip()
            allSamples.append(original_content)
            # 标注之后的序列
            sy = ['O'  for i in range(len(original_content))]
        with codecs.open(label_filename,encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                lineList=line.split('\t')
                start,end,label=int(lineList[1]),int(lineList[2]),lineList[3].replace('\r\n','')
                entity=original_content[start:end]
                # 判断实体的长度 根据长度已经实体类型进行标注
                if len(entity)==1:
                    sy[start]='S-'+label_dict.get(label)
                if len(entity)==2:
                    sy[start]='B-'+label_dict.get(label)
                    sy[start+1]='E-'+label_dict.get(label)
                else:
                    sy[start]='B-'+label_dict.get(label)
                    sy[end-1]='E-'+label_dict.get(label)
                    for  i in range(start+1,end-1):
                        sy[i]='M-'+label_dict.get(label)

        allSample_labels.append(sy)
        posFile=file.replace('.txtoriginal','-label')
        with open(posFile,'w',encoding='utf-8') as f:
            for x,y in zip(original_content,sy):
                f.write(x+'\t'+y)
                f.write('\n')

        for x, y in zip(original_content, sy):
            fw.write(x + '\t' + y)
            fw.write('\n')
    return allSamples,allSample_labels

class LSTM_CRF(object):
    def __init__(self,rnn_size=10,embedding_size=50, max_sequence_length=20,num_tags=6,batch_size=30):
        self.rnn_size=rnn_size
        self.embedding_size=embedding_size
        self.max_sequence_length=max_sequence_length
        self.num_tags=num_tags
        self.batch_size=batch_size

    def lstm_model(self):
        self.x=tf.placeholder(tf.int32,[None,self.max_sequence_length])
        self.y=tf.placeholder(tf.int32,[None,self.max_sequence_length])
        self.dropout_keep_prob=tf.placeholder(tf.float32)

        embedding_mat=tf.Variable(tf.random_uniform((max_id+1,self.embedding_size),-1.0,1.0))
        print('embedding_mat:',embedding_mat)
        embedding_output=tf.nn.embedding_lookup(embedding_mat,self.x)
        cell=tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_size)
        output,state= tf.nn.dynamic_rnn(cell,embedding_output,dtype=tf.float32)
        self.output=tf.nn.dropout(output,self.dropout_keep_prob)

    def cfr_model(self):
        # All the squences in this example have the same length,but they can be variable in a real model
        sequence_length = np.full(self.batch_size, self.max_sequence_length, dtype=np.int32)
        weights = tf.get_variable('weights', [self.rnn_size, self.num_tags],dtype=tf.float32)
        bias = tf.get_variable('bias',[self.num_tags],dtype=tf.float32)
        output_ = tf.reshape(self.output, [-1, self.rnn_size])
        print('output:', output_)
        print('weights:', weights)
        # sequence_lengh=[19 19 19 19 19 19 19 19 19 19]

        # Train and evaluate the model
        #with tf.reset_default_graph():
        with tf.Session() as session:
            # Add the data the the tensorflow graph
            # y_t = tf.convert_to_tensor(y_data)
            # print('y_t:',y_t)
            sequence_lengths_t = tf.constant(sequence_length)
            # compute unary scores from a linear layer
            matricized_unary_scores = tf.matmul(output_, weights)+bias
            unary_scores = tf.reshape(matricized_unary_scores, [-1, self.max_sequence_length, self.num_tags])
            print('unary_scores:', unary_scores)
            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                unary_scores, self.y, sequence_lengths_t)
            # Compute the verterbi sequences and scores
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                unary_scores, transition_params, sequence_lengths_t
            )
            # add a train op to turn the parameters
            loss = tf.reduce_mean(-log_likelihood)
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            session.run(tf.global_variables_initializer())
            # mask = (np.expand_dims(np.arange(max_sequence_length), axis=0)
            #         < np.expand_dims(sequence_length, axis=1))
            total_labels = np.sum(sequence_length)
            print('total_labels:',total_labels)
            print('len(x_data):',len(x_data))

            # Train a fixed number of iteration
            for i in range(200):
                x_batch,y_batch=get_batch(self.batch_size)
                tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op],feed_dict={self.x:x_batch,self.y:y_batch,self.dropout_keep_prob:0.5})
                if i % 1 == 0 or i == 100:
                    # correct_labels = np.sum((y_data== tf_viterbi_sequence) * mask)
                    #print('y_data==tf_viterbi_sequence:',y_batch==tf_viterbi_sequence)
                    correct_labels=np.sum((y_batch==tf_viterbi_sequence))
                    print('correct_labels:',correct_labels)
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    print('Accuracy:%.2f%%' % accuracy)


def sample2ids(corpus,allSamples):
    # id:char
    id2char={id:char for id,char in enumerate(corpus)}
    # char:id
    char2id={char:id for id,char in id2char.items()}
    print(len(char2id),len(id2char))

    # 将每个样本转换为id sequence
    idsamples=[]
    for sample in allSamples:
        idsamples.append([char2id.get(char) for char in sample])
    return idsamples

def regular_data(x_data,y_data):
    #print(x_data)
    # 先对y_data进行处理 以满足vocab_proccessor的要求
    labels=[]
    for row in y_data:
        s=''
        for item in row:
            s=' '.join(item)
        labels.append(s.strip())

    vocab_processor=tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,min_frequency=1)
    text_processed=np.array(list(vocab_processor.fit_transform(x_data)))

    # 计算text_processed中最大的标号 即为单词的总个数
    max_id=max([item for row in text_processed for item in row])
    print('max_id:',max_id)

    vocab_processor=tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,min_frequency=1)
    label_processed=np.array(list(vocab_processor.fit_transform(labels)))
    print('label_processed:',label_processed)
    print('text_processed:',text_processed)
    return text_processed,label_processed,max_id

def get_batch(batch_size):
    ids=np.random.permutation(len(x_data))
    x_shuffled=x_data[ids]
    y_shuffled=y_data[ids]
    return x_shuffled[:batch_size],y_shuffled[:batch_size]
if __name__ == '__main__':
    # 定义全局变量
    max_sequence_length=20
    #第一步：标注数据 并写入一个文件中
    basedir=os.path.join(os.getcwd(),'train_data600')
    pattern='*.txtoriginal.txt'
    all_txtoriginal_texts=glob.glob(os.path.join(basedir,pattern))
    allSamples,allSample_labels=labelChar(all_txtoriginal_texts)
    #x_data=sample2ids(corpus,allSamples)
    #将x_data,y_data规整为统一长度的样本集
    x_data,y_data,max_id=regular_data(allSamples,allSample_labels)

    # # 第二步：创建LSTM模型
    lstm_crf=LSTM_CRF(10,50)
    lstm_crf.lstm_model()
    lstm_crf.cfr_model()





