# -*- coding: utf-8 -*-
"""
 @Time    : 2017/12/29 0029 上午 9:42
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:使用textRank进行摘取式文本摘要的生成
"""
import nltk
from  nltk.corpus import stopwords
from nltk.cluster.util import  cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    file=open(file_name,'r')
    filedata=file.readlines()
    articles=filedata[0].split('. ')
    sentences=[]

    for sentence in articles:
        print(sentence)
        sentences.append(sentence.replace('[^a-zA-Z]',' ').split(' '))
    sentences.pop()
    return sentences

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    sent1=[w.lower() for w in sent1]
    sent2=[w.lower() for w in sent2]

    all_words=list(set(sent1+sent2))
    vector1=[0]*len(all_words)
    vector2=[0]*len(all_words)

    #build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)]+=1
    #build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)]+=1
    return 1-cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stop_words):
    #Create an empty similarity matrix
    similarity_matrix=np.zeros((len(sentences),len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2: #自身与自身
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2])
    return similarity_matrix

def generate_summary(file_name,top_n):
    nltk.download('stopwords')
    stop_words=stopwords.words('english')
    summarize_text=[]

    #step1:Read text and split it
    sentences=read_article(file_name)

    #step2:Generate similary matrix across sentences
    sentence_similarity_matrix=build_similarity_matrix(sentences,stop_words)

    #step3:Rank sentneces in simlarity matrix
    sentences_similarity_graph=nx.from_numpy_array(sentence_similarity_matrix)
    scores=nx.pagerank(sentences_similarity_graph)

    #step4: sort the rank and pick top sentences
    ranked_sentences=sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    print('Indexes of top ranked_sentence order are:',ranked_sentences)

    for i in range(top_n):
        summarize_text.append(' '.join(ranked_sentences[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

if __name__ == '__main__':
    generate_summary('mstf.txt',2)