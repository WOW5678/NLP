# -*- coding:utf-8 -*-
'''

'''
import re
import numpy as np

sents=open('data/msr_training_test.utf8',encoding='utf-8').read().strip()
print('sents:',repr(sents))
sents=sents.split('\n') # 这个语料库的换行方式是\n
# 注意‘ +’是个正则表达式，是匹配至少一个空格
sents=[re.split(' +',s) for s in sents] # 词之间以空格隔开
#print('sents_2:',sents)
sents=[[w for w in s if w] for s in sents] # 去掉空字符串
#print('sents:',sents)
np.random.shuffle(sents) #打乱预料，以便后面划分验证集

#字符计数
chars={} # 统计字表
for s in sents:
    for c in ''.join(s):
        if c in chars:
            chars[c]+=1
        else:
            chars[c]=1
min_count=2 # 过滤低频词
chars={i:j for i,j in chars.items() if j>=min_count}

# id到字的映射  id=0用于那些不存在该字符集中的字符
id2char={i+1:j for i,j in enumerate(chars)}
#字到id的映射
chars2id={j:i for i,j in id2char.items()}

id2tag={0:'s',1:'b',2:'m',3:'e'} #标签（sbme）与id之间的映射
tag2id={j:i for i,j in id2tag.items()}

# 最后的5个样本作为验证集
train_sents=sents[:-5]
valid_sents=sents[-5:]

from keras.utils import to_categorical
batch_size=10
# 该函数作用：每次生成一小批数据用于后续的训练，以减少对显存的占用
def train_generator():
    while True:
        X,Y=[],[]
        for i,s in enumerate(train_sents):
            #sx是每个字符序列 sy是每个字符对应的标注
            sx,sy=[],[]
            for w in s: #遍历句子中每个词
                sx.extend([chars2id.get(c,0) for c in w])# 遍历词中的每个字符
                if len(w)==1:
                    sy.append(0) # 单字词的标签（s）
                elif len(w)==2:
                    sy.extend([1,3]) #双字词的标签(b,e)
                else:
                    sy.extend([1]+[2]*(len(w)-2)+[3]) #多于两字符的标签
            X.append(sx)
            Y.append(sy)
            if len(X)==batch_size or i==len(train_sents)-1: # 如果达到一个batch或者是最后一个不满的batch
                maxlen=max(len(x) for x in X) # 找出这个batch中句子的最大字符数
                X=[x+[0]*(maxlen-len(x)) for x in X] #不足则补0
                print('X[0].length:',len(X[0]))
                Y=[y+[4]*(maxlen-len(y)) for y in Y] # 不足则补第5个标签
                yield np.array(X),to_categorical(Y,5)

from crf_keras import CRF
from keras.layers import Dense,Embedding,Conv1D,Input
from keras.models import Model

embedding_size=100
sequence=Input(shape=(None,),dtype='int32') #建立输入层，输入长度设为None
embedding=Embedding(len(chars)+1,embedding_size)(sequence) # 去掉了mask_zeros=True

cnn=Conv1D(32,3,activation='relu',padding='same')(embedding)
cnn=Conv1D(32,2,activation='relu',padding='same')(cnn)
cnn=Conv1D(32,3,activation='relu',padding='same')(cnn) # 层叠3层cnn
crf=CRF(True) #定义crf层，参数为True,自动mask掉最优的一个标签
tag_score=Dense(5)(cnn) # 变成了5分类，弟5个标签用来mask掉
print('tag_score:',tag_score)
# 会顺序执行build函数 和call函数
tag_score=crf(tag_score) #包装一下原来的tag_score
print('tag_score after crf:',tag_score)

model=Model(inputs=sequence,outputs=tag_score)
model.summary()

model.compile(loss=crf.loss,
              optimizer='adam',
              metrics=[crf.accuracy])

#定义一个求字典中最大值的函数
def max_in_dict(d):
    key,value=list(d.items())[0]
    for i,j in list(d.items())[1:]:
        if j>value:
            key,value=i,j
        return key,value
# 维特比算法，与HMM一致
def vertibi(nodes,trans):
    paths=nodes[0]
    for l in range(1,len(nodes)):
        paths_old,paths=paths,{}
        for n,ns in nodes[l].items(): #当前时刻的所有节点
            max_path,max_score='',-1e10
            for p,ps in paths_old.items():
                # 为什么是加而不是乘呢？
                #逐标签得分，加上转移概率得分?
                score=ns+ps+trans[p[-1]+n] #计算新分数
                if score>max_score:
                    max_path,max_score=p+n,score #更新路径
            paths[max_path]=max_score
    return max_in_dict(paths)

# 分词函数，也跟前面的HMM基本一致
def cut(s,trans):
    #空字符直接返回
    if not s:
        return []
    #get(key,char):当key不存在时，返回char
    # 字符序列转换为id序列，经过我们前面对预料的处理，处理后的字符集是没有空格的，
    #所以这里简单的将空格的id跟句号的id等同起来
    sent_ids=np.array([[chars2id.get(c,0) if c!=' ' else chars2id[u'。']
                        for c in s]])
    probas=model.predict(sent_ids)[0] #预测模型
    print('probas:',probas)
    nodes=[dict(zip('sbme',i)) for i in probas[:,:4]]#只取前4个
    print('nodes:',nodes)
    nodes[0]={i:j for i,j in nodes[0].items() if i in 'bs'}#首字标签只能是b或者s
    nodes[-1]={i:j for i,j in nodes[-1].items() if i in 'es' }#末字标签只能是e或者s
    tags=vertibi(nodes,trans)[0]
    result=[s[0]]
    for i,j in zip(s[1:],tags[1:]):
        if j in 'bs':
            result.append(i)
        else:
            result[-1]+=i
    return result

from keras.callbacks import Callback
from tqdm import tqdm

#自定义callback类
class Evaluate(Callback):
    def __init__(self):
        self.highest=0.
    def on_epoch_end(self,epoch,logs=None):
        _=model.get_weights()[-1][:4,:4] # 从训练模型中取出最新得到的转移矩阵
        trans={}
        for i in 'sbme':
            for j in 'sbme':
                trans[i+j]=_[tag2id[i],tag2id[j]]
        right=0.0
        total=0.0
        #可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
        for s in tqdm(iter(valid_sents),desc=u'验证模型中'):
            result=cut(''.join(s),trans)
            total+=len(set(s))
            # 直接将词集的交集作为正确数，该指标比较简单，
            # 也许会导致估计偏高，读者可以自己考虑自定义指标
            right+=len(set(s)& set(result))
        acc=right/total
        if acc>self.highest:
            self.highest=acc
        print('val acc:%s,highest:%s'%(acc,self.highest))

# 建立Callback类
evaluator=Evaluate()
#训练并将evaluate加入到训练过程
# steps_per_epoch：表示一个epoch分成多少个batch_size
model.fit_generator(train_generator(),steps_per_epoch=100,epochs=2,callbacks=[evaluator])




