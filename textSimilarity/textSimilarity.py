# -*- coding:utf-8 -*-
#功能：不同的词向量化方法（词袋，tf-idf,lSI）并计算文本的相似性

import jieba.posseg as pseg
import codecs
from gensim import corpora, models,similarities
'''
获取停用词列表
'''
def getStopWords(filename):
    stopwords = codecs.open(filename, 'r').readlines()
    stopwords = [w.strip() for w in stopwords]
    return stopwords
'''
对一篇文章进行分词，去停用词 返回处理后的单词
'''
def tokenization(filename):
    result=[]
    with open(filename) as f:
        text=f.read()
        #分词
        words=pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopWords:
            result.append(word)
    return result

if __name__ == '__main__':
    stopWords=getStopWords('stop_words.txt')
    #结巴分词的停用词性
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    filenames=['高血压1.txt','高血压2.txt','安卓手机.txt']
    corpus=[]
    for each in filenames:
        corpus.append(tokenization(each))
    print(corpus)

    #建立词袋模型
    dictionary=corpora.Dictionary(corpus)
    print('dictionary:',len(dictionary))
    doc_vectors=[dictionary.doc2bow(text) for text in corpus]
    print(len(doc_vectors))
    #每个元祖中第一个为索引 第二个为词频
    print('doc_vectors:',doc_vectors)

    #建立tf-idf模型
    tfidf=models.TfidfModel(doc_vectors)
    tfidf_vectors=tfidf[doc_vectors]
    print(len(tfidf_vectors))
    print('tfidf_vectors:',len(tfidf_vectors[0]))

    #构建一个查询文本，是高血压主题的，利用词袋模型的字典将其映射到向量空间
    query=tokenization('高血压查询.txt')
    query_bow=dictionary.doc2bow(query)

    print('len(query_bow):',len(query_bow))
    print('query_bow:',query_bow)

    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]
    print (list(enumerate(sims)))


    #构建LSI模型，设置主题数为2（理论上这两个主题应该分别为高血压和安卓手机）
    lsi=models.LsiModel(tfidf_vectors,id2word=dictionary,num_topics=2)
    print(lsi.print_topics(2))

    lsi_vector=lsi[tfidf_vectors]
    for vec in lsi_vector:
        print('vec:',vec)
    #在lsi向量空间，所有文本的向量都是二维的
    query=tokenization('高血压查询.txt')
    query_bow=dictionary.doc2bow(query)
    print(query_bow)

    query_lsi=lsi[query_bow]
    print('query_lsi:',query_lsi)

    index=similarities.MatrixSimilarity(lsi_vector)
    sims=index[query_lsi]
    print(list(enumerate(sims)))




