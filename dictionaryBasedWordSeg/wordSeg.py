# -*- coding: utf-8 -*-
"""
 @Time    : 2018/8/11 0011 下午 4:43
 @Author  : Shanshan Wang
 @Version : Python3.5
function:
实现三种基于匹配的分词算法（即基于词典的分词方法），分别为前向最大匹配（FMM）
后向最大匹配（BMM）和双向最大匹配（DBMaxMatch）
使用的词库（字典）是搜狗互联网词库，http://www.sogou.com/labs/resource/w.php
"""
def get_text(file):
    with open(file,'r',encoding='utf-8') as f:
        return f.read().replace('\ufeff','')

def get_wrodDict(file):
    wordDict=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            wordDict.append(line.split('\t')[0])
    return wordDict

class FMM(object):
    def __init__(self,sentence,wordDict):
        self.sentence=sentence
        self.wordDict=wordDict
        self.result=[]
    def word_segmentation(self,maxLength):
        self.maxLength=maxLength
        while self.sentence:
            self.maxLength=min(len(self.sentence),self.maxLength)
            while True:
                # 按照当前长度切出一个词
                word=self.sentence[0:self.maxLength]
                # 如果当前切除的词在字典中，注意：单个汉字被认为总是在字典中
                # 则切走当前词，并更新sentence,进行下一轮的切分
                if len(word)==1 or word in self.wordDict:
                    self.result.append(word)
                    self.sentence=self.sentence[self.maxLength:]
                    self.maxLength=maxLength
                    break
                else:
                    #当前词不在字典中，则减小切分长度，重新尝试切割
                    self.maxLength-=1
        return self.result

class BMM(object):
    def __init__(self,sentence,wordDict):
        self.sentence=sentence
        self.wordDict=wordDict
        self.result=[]
    def word_segmentation(self,maxLength):
        self.maxLength=maxLength
        while self.sentence:
            self.maxLength=min(len(self.sentence),self.maxLength)
            while True:
                # 按照当前长度切出一个词(从后向前)
                word=self.sentence[-self.maxLength:]
                # 如果当前切除的词在字典中，注意：单个汉字被认为总是在字典中
                # 则切走当前词，并更新sentence,进行下一轮的切分
                if len(word)==1 or word in self.wordDict:
                    self.result.append(word)
                    self.sentence=self.sentence[:-self.maxLength]
                    self.maxLength=maxLength
                    break
                else:
                    #当前词不在字典中，则减小切分长度，重新尝试切割
                    self.maxLength-=1
        # 因为是从后往前进行扫描的，所有需要对列表进行翻转操作
        self.result.reverse()
        return self.result

class DBMaxMatch(object):
    def __init__(self,sentence,wordDict):
        self.sentence=sentence
        self.wordDict=wordDict
    def word_segmentation(self,maxLength=6):
        FMM_res=FMM(self.sentence,self.wordDict).word_segmentation(maxLength)
        BMM_res=BMM(self.sentence,self.wordDict).word_segmentation(maxLength)

        FMM_in_dict=[word for word in FMM_res if word in self.wordDict]
        FMM_not_in_dict=[word for word in FMM_res if word not in self.wordDict]
        FMM_single=[word for word in FMM_res if len(word)==1]

        BMM_in_dict=[word for word in BMM_res if word in self.wordDict]
        BMM_not_in_dict=[word for word in BMM_res if word not in self.wordDict]
        BMM_single=[word for word in BMM_res if len(word)==1]

        same_res=True
        if len(FMM_res)!=len(BMM_res):
            same_res=False
        else:
            for i in range(len(FMM_res)):
                if FMM_res[i]!=BMM_res[i]:
                    same_res=False
                    break

        # 1. FMM与BMM结果完全相同，返回任意一个结果即可
        if same_res:
            return FMM_res
        # 2.返回不在词典中词数少的那个结果
        elif len(FMM_not_in_dict)!=len(BMM_not_in_dict):
            if len(FMM_not_in_dict)<len(BMM_not_in_dict):
                return FMM_res
        # 3.返回单字数少的结果
        elif len(FMM_single)!=len(BMM_single):
            if len(FMM_single)<len(BMM_single):
                return FMM_res
        # 4.都相同的话，返回BMM的结果
        else:
            return BMM_res

if __name__ == '__main__':
    content=get_text('text.txt')
    wordDict=get_wrodDict('sogouW_Freq/SogouLabDic.dic')
    fmm=FMM(content,wordDict)
    segmentation_result=fmm.word_segmentation(maxLength=6)
    print(segmentation_result)
    bmm=BMM(content,wordDict)
    segmentation_result=bmm.word_segmentation(maxLength=6)
    print(segmentation_result)
    dbMaxMatch=DBMaxMatch(content,wordDict)
    result=dbMaxMatch.word_segmentation(maxLength=6)
    print(result)
