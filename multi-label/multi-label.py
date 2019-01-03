# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/2 0002 上午 8:43
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 实现一个多标签分类的小案例
"""
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_path='data/train.csv'
data_raw=pd.read_csv(data_path)
print(data_raw.shape)
print('Number of rows in data=',data_raw.shape[0])
print('Number of columns in data=',data_raw.shape[1])
print('\n')
print(data_raw.head(3))

# Checking for missing values
missing_values_check=data_raw.isnull().sum()
print(missing_values_check)

#计算每个标签下的评论数
#标签为0的评论被看做是clean的评论，创建分割的列来识别clean评论
# 使用axis=1计数row-wise and axis=0来计数column wise
rowSums=data_raw.iloc[:,2:].sum(axis=1)
print('rowSums:',rowSums)
clean_comments_count=(rowSums==0).sum(axis=0)
print('Total number of comments=',len(data_raw)) #159571
print('Number of clean comments=',clean_comments_count)#143346
print('Number of comments with labels=',len(data_raw)-clean_comments_count)#16225

categories=list(data_raw.columns.values)
categories=categories[2:]
print(categories) #['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#计算在每个类别下的评论个数
counts=[]
for category in categories:
    counts.append((category,data_raw[category].sum()))
df_stats=pd.DataFrame(counts,columns=['category','Number of comments'])
print('df_stats:',df_stats)

#画图
sns.set(font_scale=2)
plt.figure(figsize=(15,8))
ax=sns.barplot(categories,data_raw.iloc[:,2:].sum().values)
plt.title('Comments in each category',fontsize=24)
plt.ylabel('Number of comments',fontsize=18)
plt.xlabel('Comment type',fontsize=18)

#adding the text label
rects=ax.patches
labels=data_raw.iloc[:,2:].sum().values
for rect,label in zip(rects,labels):
    height=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2,height+5,label,ha='center', va='bottom', fontsize=18)
plt.show()

#计算有多个标签的评论个数
rowSums=data_raw.iloc[:,2:].sum(axis=1)
multiLabel_counts=rowSums.value_counts()
print('multiLabel_counts:',multiLabel_counts)
multiLabel_counts=multiLabel_counts.iloc[1:]

sns.set(font_scale=2)
plt.figure(figsize=(15,8))
ax=sns.barplot(multiLabel_counts.index,multiLabel_counts.values)
plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)

#adding the text labels
rects=ax.patches
labels=multiLabel_counts.values
for rect,label in zip(rects,labels):
    height=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2,height+5,label,ha='center',va='bottom')
plt.show()

#使用词云来表现每个类别下最频繁出现的单词
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))

#toxic
subset=data_raw[data_raw.toxic==1]
text=subset.comment_text.values
cloud_toxic=WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=2500,
    height=1800
).generate(' '.join(text))

plt.subplot(2,3,1)
plt.axis('off')
plt.title('Toxic',fontsize=40)
plt.imshow(cloud_toxic)

#severe_toxic
subset = data_raw[data_raw.severe_toxic==1]
text = subset.comment_text.values
cloud_severe_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 2)
plt.axis('off')
plt.title("Severe Toxic",fontsize=40)
plt.imshow(cloud_severe_toxic)

# obscene
subset = data_raw[data_raw.obscene==1]
text = subset.comment_text.values
cloud_obscene = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 3)
plt.axis('off')
plt.title("Obscene",fontsize=40)
plt.imshow(cloud_obscene)

# threat
subset = data_raw[data_raw.threat==1]
text = subset.comment_text.values
cloud_threat = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 4)
plt.axis('off')
plt.title("Threat",fontsize=40)
plt.imshow(cloud_threat)

# insult
subset = data_raw[data_raw.insult==1]
text = subset.comment_text.values
cloud_insult = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 5)
plt.axis('off')
plt.title("Insult",fontsize=40)
plt.imshow(cloud_insult)

# identity_hate
subset = data_raw[data_raw.identity_hate==1]
text = subset.comment_text.values
cloud_identity_hate = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 6)
plt.axis('off')
plt.title("Identity Hate",fontsize=40)
plt.imshow(cloud_identity_hate)
plt.show()

# 2. Data Pre-Processing
data=data_raw
data=data_raw.loc[np.random.choice(data_raw.index,size=2000)]
print(data.shape) #(2000, 8)

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
# 2.1 cleaning data
def cleanHtml(sentence):
    #.:除换行符以外的任意字符
    #*：匹配前面的子表达式零次或多次
    #？：匹配前面的子表达式零次或一次。等价于{0,1}
    cleanr=re.compile('<.*?>')
    cleantext=re.sub(cleanr,' ',str(sentence))
    return cleantext

#function to clean the word of any punctuation or special characters
#移除特殊字符
#还有别的简单的实现方法，比如使用string.punctions
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

#其实可以使用isalpha()函数直接实现，
#这种方法有点复杂了
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
print(data.head(5))

# 2.2 Removing Stop Words
stop_words=set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
data['comment_text'] = data['comment_text'].apply(removeStopWords)
print(data.head(5))

# 2.3 Stemming
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['comment_text'] = data['comment_text'].apply(stemming)
print(data.head(5))

# 2.4 Train-Test Split
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)
print(train.shape)
print(test.shape)
train_text = train['comment_text']
test_text = test['comment_text']

# 2.5 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)
x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['id','comment_text'], axis=1)
x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['id','comment_text'], axis=1)

# 3. Multi-Label Classification
# 3.1. Multiple Binary Classifications - (One Vs Rest Classifier)¶
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

for category in categories:
    print('**Processing {} comments...**'.format(category))

    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])

    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n")

#3.2 Multiple Binary Classifications - (Binary Relevance)
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")

# 3.3 Classifier Chains
# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression

# initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())

# Training logistic regression model on train data
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")

# 3.4 Label Powerset
from skmultilearn.problem_transform import LabelPowerset
classifier = LabelPowerset(LogisticRegression())

# train
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")

#3.5. Adapted Algorithm
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
classifier_new = MLkNN(k=10)

# Note that this classifier can throw up errors when handling sparse matrices.

x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()

# train
classifier_new.fit(x_train, y_train)

# predict
predictions_new = classifier_new.predict(x_test)

# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions_new))
print("\n")
