# -*- coding:utf-8 -*-
'''
Create time: 2020/5/23 11:14
@Author: 大丫头
'''

import spacy
from collections import Counter
from string import punctuation

# load spacy model

#方式1

# python -m spacy download en_core_web_lg
nlp=spacy.load('en_core_web_lg')

#方式2
# import en_core_web_lg
# nlp=en_core_web_lg.load()


# 关键字提取代码
def get_hotwords(text):
    results=[]
    # 专有名词 形容词 名词
    pos_tags=['PROPN','ADJ','NOUN']
    doc=nlp(text.lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tags:
            results.append(token.text)

    return results

# 测试函数
output=get_hotwords('''Welecome to Medium! Medium is a publishing platform.''')
print(output)

# 去除重复的单词
output=set(get_hotwords('''Welecome to Medium! Medium is a publishing platform.'''))
print(output)

# 从keywords中生成hashtags
output=set(get_hotwords('''Welecome to Medium! Medium is a publishing platform.'''))
print(output)
hashtags=[('#'+x) for x in output]
print(' '.join(hashtags))

# 通过frequency对keywords进行排序
output=get_hotwords('''Welecome to Medium! Medium is a publishing platform.''')
print(output)
hashtags=[('#'+x) for x in Counter(output).most_common(5)]
print(' '.join(hashtags))