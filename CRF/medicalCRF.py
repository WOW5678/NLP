# -*- coding:utf-8 -*-
'''
function:
1. 处理medical data,根据标记文件 标注数据
'''
import codecs
import os
import glob
import pycrfsuite
# 使用SMBE方法进行数据标注
def labelChar(all_txtoriginal_texts):
    fw=open('labelData.txt', 'w')
    label_dict={'解剖部位':'body','手术':'surgery','药物':'drug',
                '独立症状':'ind_symptoms','症状描述':'SymptomDes'}
    for file in all_txtoriginal_texts:
        original_filename=file
        label_filename=file.replace('txtoriginal.','')
        print(label_filename)
        with codecs.open(original_filename,encoding='utf-8') as f:
            original_content=f.read().strip()
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
        for x, y in zip(original_content, sy):
            fw.write(x + '\t' + y)
            fw.write('\n')


class CRF(object):
    def __init__(self,c1=1.0,c2=1e-3,max_iterations=500,feature_possible_transitions=True,feature_minfre=3):
        self.c1=c1,
        self.c2=c2,
        self.max_iterations=max_iterations,
        self.feature_possible_transitions=feature_possible_transitions,
        self.feature_minfreq=feature_minfre

    # Define features
    # i is the index of one sentence ,sents is all the lines
    def word2features(self,i,sents,labels):
        word = sents[i]
        print('word:',word)
        postag =labels[i]
        print('postag:',postag)

        self.features = [
            'bias',
            'word=' + word,
            'word_tag=' + postag
        ]
        if i > 0:
            self.features.append('word[-1]=' + sents[i - 1])
            self.features.append('word[-1]_tag=' + sents[i - 1])
            if i > 1:
                self.features.append('word[-2]=' + sents[i - 2])
                self.features.append('word[-2]_tag=' + sents[i - 2])
        if i < len(sents) - 1:
            self.features.append('word[1]=' + sents[i + 1])
            self.features.append('word[1]_tag=' + sents[i + 1])
            if i < len(sents) - 2:
                self.features.append('word[2]=' + sents[i + 2])
                self.features.append('word[2]_tag=' + sents[i + 2])
        return self.features

    def train(self,x_train,y_train):
        model = pycrfsuite.Trainer(verbose=True)
        print(x_train)
        model.append(x_train, y_train)
        model.set_params({
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations,
            'feature.possible_transitions': self.feature_possible_transitions,
            'feature.minfreq': self.feature_minfreq
        })
        model.train('./medical.crfsuite')

def getSentes_labels(filename):
    with open(filename) as f:
        lines=f.readlines()
        sents_=[line.split('\t')[0] for line in lines]
        labels_=[line.split('\t')[1].replace('\n','') for line in lines]
    return sents_,labels_

if __name__ == '__main__':
    '''
    #第一步：标注数据 并写入一个文件中
    basedir=os.path.join(os.getcwd(),'train_data600')
    pattern='*.txtoriginal.txt'
    all_txtoriginal_texts=glob.glob(os.path.join(basedir,pattern))
    labelChar(all_txtoriginal_texts)
    '''
    # 第二步：根据数据生成特征和标签
    sents,labels=getSentes_labels('test.txt')
    print(type(sents), type(labels))
    print('labels:',labels)
    print('sents:',sents)
    crf=CRF()
    X_train= [crf.word2features(i,sents,labels) for i in range(len(sents))]
    crf.train(X_train,labels)