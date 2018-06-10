# -*- coding:utf-8 -*-
#功能：利用马尔可夫链实现随机文本生成

import codecs
import jieba
import random
import re

class Markov:
    def __init__(self,filepath=None,mode=0,coding="utf-8"):
        self.dicLen=0 #前缀字典长度
        self.Cap=[]     #可作为语句开始词语的集合
        self.capLen=0   #可作为语句开头词语的集合长度的
        self.mode=mode  #模式 中文为1 英文为0
        self.coding=coding #编码方式
        self.dic={}       #前缀字典
        if filepath is not None:
            self.train(filepath,self.mode,coding)
    def train(self,filepath='',mode=0, coding="utf-8"):
        self.dic={}
        self.Cap=[]
        self.mode=mode
        self.coding=coding
        if filepath is None or filepath=='':
            return
        eg_puncmark=re.compile('[\,\.\!\;\:\?\~\`\#\$\%\@\*\(\)\]\[]')  #英文标点正则
        zh_puncmark=re.compile('[,。！；]')                              #中文标点正则
        with codecs.open(filepath,"r",coding) as f:
            for line in f.readlines():
                words=[]
                line=re.sub('[\r\n]',"",line) #清除行末的回车
                if mode==0:#如果是英文模式 使用eg_puncmark符号进行分割
                    sentences=eg_puncmark.split(line)
                    sentences_words=[]
                    for sentence in sentences:
                        #按空格划分单词并过滤空串
                        sentences_words.append(filter(lambda x:x!='',sentence.strip(" ")))
                    for word in sentences_words:                          #对每句中的单词
                        for i in range(len(words)-2):
                            #将两个连续的词拼接为前缀
                            keypair=word[i]+" "+word[i+1]
                            if keypair[0].isUpper():
                                #若前缀的首字符为大写 则添加到可作为开头的前缀集合中
                                self.Cap.append(keypair)
                            if self.dic.get(keypair) is None:
                                self.dic[keypair]=[words[i+2]]
                            else:
                                #已经存在于前缀集合中  则为这个前缀对应的列表中添加成员
                                self.dic[keypair].append(words[i+2])

                #中文模式下的训练数据
                else:
                    sentences=zh_puncmark.split(line)
                    for sentence in sentences: #对于每个句子
                        jwords=jieba.cut(sentence,cut_all=False) #使用jieba词库进行中文分词
                        for word in jwords: #对于分割后的每个词组 进考虑长度大于2的中文词语
                            if len(word)>=2:
                                words.append(word)
                        if len(words)>2:
                            self.Cap.append(words[0]+" "+words[1]) #添加该每句开头的两个单词
                            words=list(filter(lambda x:x!='',words) )    #过滤空串
                            #print (words)
                            for i in range(len(words)-2):
                                keypair=words[i]+" "+words[i+1]    #组建前缀
                                if self.dic.get(keypair) is None:
                                    self.dic[keypair]=[words[i+2]]
                                else:
                                    #为该前缀添加后缀
                                    self.dic[keypair].append(words[i+2])
        #更新前缀的字典长度
        self.dicLen=len(self.dic)
        self.capLen=len(self.Cap)
    #来观察生成的前缀字典
    def getDic(self):
        return self.dic

    #随机语句生成函数
    def say(self,length=10):
        print (self.dicLen)
        if self.dicLen<=2:  #如果前缀字典的长度小于等于2 则认为很难构成一个可行的句子
            print("I feel tired and I need food to say something.")

        else:
            keypair=self.Cap[random.randint(0, self.capLen-1)] #随机选取可以作为语句开头的前缀
            fst,snd=keypair.split(" ")[0],keypair.split(" ")[1] #将前缀按照空格拆分成词语
            global len
            pairLen=len(self.dic[keypair])                     #keypair作为前缀时有多少匹配项（后缀）
            if self.mode==0:
                sentence=fst+" "+snd
            else:
                sentence=fst+snd  #中文模式下不需要空格
            #只有当 要生成的句子长度不大于0了(不可能发生) 或者当前词对应的后缀项为空了 则结束循环
            while length>0 and pairLen>0:
                temp=self.dic[keypair][random.randint(0,pairLen-1)] #随机选取后缀词语
                fst=snd
                snd=temp
                if self.mode==0:
                    sentence=sentence+" "+snd
                else:
                    sentence=sentence+snd
                keypair=fst+" "+snd                  #更新待搜索前缀
                if self.dic.get(keypair) is not None:
                    pairLen=len(self.dic[keypair])
                else:   #当前单词不在前缀字典中  即当前词没有后缀词语
                    break
                length-=1
            #退出while 循环之后 给句子加上末尾标点
            if self.mode==0:
                print (sentence+".")
            else:

                print (sentence+"。")


if __name__ == '__main__':
    markov=Markov("swords.txt",mode=1)
    #print (markov.getDic())
    markov.say(10)

