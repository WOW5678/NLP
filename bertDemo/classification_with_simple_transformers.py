# -*- coding:utf-8 -*-
'''
Create time: 2020/6/8 14:21
@Author: 大丫头
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import simpletransformers
from simpletransformers.classification import ClassificationModel
import sklearn
import itertools
import emoji

### 预处理数据
df=pd.read_csv('./train.csv')
column_todrop=['keyword','location']
df.drop(columns=column_todrop,inplace=True)
print(df.head())

fake_tweet=df[df.target==0]
print(fake_tweet.shape) #[4342,3]

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"thx"   : "thanks"
}

# 将以上的简写都转变成全写的形式
def remove_contractions(text):
    return contractions[text.lower()] if text.lower() in contractions.keys() else text

df['text']=df['text'].apply(remove_contractions)
print(df.head())

## clean the dataset
def clean_dataset(text):
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    # Remove HTML special entities (e.g. &amp;)
    text = re.sub(r'\&\w*;', '', text)
    # Remove tickers
    text = re.sub(r'\$\w*', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    # Remove whitespace (including new line characters)
    text = re.sub(r'\s\s+','', text)
    text = re.sub(r'[ ]{2, }',' ',text)
    # Remove URL, RT, mention(@)
    text=  re.sub(r'http(\S)+', '',text)
    text=  re.sub(r'http ...', '',text)
    text=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','',text)
    text=  re.sub(r'RT[ ]?@','',text)
    text = re.sub(r'@[\S]+','',text)
    # Remove words with 4 or fewer letters
    text = re.sub(r'\b\w{1,4}\b', '', text)
    #&, < and >
    text = re.sub(r'&amp;?', 'and',text)
    text = re.sub(r'&lt;','<',text)
    text = re.sub(r'&gt;','>',text)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    text= ''.join(c for c in text if c <= '\uFFFF')
    text = text.strip()
    # Remove misspelling words
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    # Remove emoji
    text = emoji.demojize(text)
    text = text.replace(":"," ")
    text = ' '.join(text.split())
    text = re.sub("([^\x00-\x7F])+"," ",text)
    # Remove Mojibake (also extra spaces)
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    return text

df['text']=df['text'].apply(clean_dataset)
print(df.head())
print(df.shape) #[7616,3]

### 将数据集分割成训练集合验证集
x_train_clean,x_test_clean,y_train_clean,y_test_clean=train_test_split(df['text'],df['target'],test_size=0.2,random_state=42)
train_df_clean=pd.concat([x_train_clean,y_train_clean],axis=1)
print("shape of training dataset:",train_df_clean.shape)
print(train_df_clean.head())
eval_df_clean=pd.concat([x_test_clean,y_test_clean],axis=1)
print("shape of training dataset:",eval_df_clean.shape)
print(eval_df_clean.head())


#### BERT MODEL TRAINING

#set up the train arguments
train_args={
    'evaluate_during_training': True,
    'logging_steps': 100,
    'num_train_epochs': 2,
    'evaluate_during_training_steps': 100,
    'save_eval_checkpoints': False,
    'train_batch_size': 32,
    'eval_batch_size': 64,
    'overwrite_output_dir': True,
    'fp16': False,
    'wandb_project': "visualization-demo"
}

model_BERT=ClassificationModel('bert','bert-base-cased',num_labels=2,use_cuda=True,cuda_device=0,args=train_args)

# 训练模型
model_BERT.train_model(train_df_clean,eval_df=eval_df_clean)

# check model performance on validation data
result,model_outputs,wrong_predictions=model_BERT.eval_model(eval_df_clean,acc=sklearn.metrics.accuracy_score)

#### 还可以使用其他的BERT模型
model_Roberta=ClassificationModel('roberta','roberta-base',use_cuda=True,cuda_device=0,args=train_args)
model_Roberta.train_model(train_df_clean,eval_df=eval_df_clean)
result,model_outputs,wrong_predictions=model_Roberta.eval_model(eval_df_clean,acc=sklearn.metrics.accuracy_score)


#### 还可以使用其他的BERT模型
model_Albert=ClassificationModel('albert','albert-base-v2',use_cuda=True,cuda_device=0,args=train_args)
model_Roberta.train_model(train_df_clean,eval_df=eval_df_clean)
result,model_outputs,wrong_predictions=model_Albert.eval_model(eval_df_clean,acc=sklearn.metrics.accuracy_score)


### Performance prediction--Test set
test_df=pd.read_csv('./test.csv')
column_todrop=['keyword','location']
test_df.drop(columns=column_todrop,inplace=True)
test_df['text']=test_df['text'].apply(remove_contractions)
test_df['text']=test_df['text'].apply(clean_dataset)


predictions,raw_outputs=model_Roberta.predict(test_df['text'])
test_df['target']=predictions
print(test_df.tail())

print(test_df['target'].value_counts())


## performance predictions on random tweets

# example 1:
test_tweet1 = "#COVID19 will spread across U.S. in coming weeks. We’ll get past it, but must focus on limiting the epidemic, and preserving lif"
test_tweet1 = remove_contractions(test_tweet1)
test_tweet1 = clean_dataset(test_tweet1)

predictions, _ = model_Roberta.predict([test_tweet1])
response_dict = {0: 'Fake', 1: 'Real'}
print("Prediction is: ", response_dict[predictions[0]])

# example 2:
test_tweet2 = "BREAKING: Confirmed flooding on NYSE. The trading floor is flooded under more than 3 feet of water."
test_tweet2 = remove_contractions(test_tweet2)
test_tweet2 = clean_dataset(test_tweet2)

predictions, _ = model_Roberta.predict([test_tweet2])
response_dict = {0: 'Fake', 1: 'Real'}
print("Prediction is: ", response_dict[predictions[0]])

# example 3:
test_tweet3 = "Everything is ABLAZE. Please run!!"
test_tweet3 = remove_contractions(test_tweet3)
test_tweet3 = clean_dataset(test_tweet3)

predictions, _ = model_Roberta.predict([test_tweet3])
response_dict = {0: 'Fake', 1: 'Real'}
print("Prediction is: ", response_dict[predictions[0]])