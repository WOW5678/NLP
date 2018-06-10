# -*- coding:utf-8 -*-
#功能：实现遗传算法  以增进对该算法的理解

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import pi, sin
import copy

class Gas(object):
    '''
    初始化有些变量
    '''
    def __init__(self,popsize,chrosize,xrangemin,xrangemax):
        self.popsize=popsize       #种群规模
        self.chrosize=chrosize     #染色体长度（编码长度）
        self.xrangemin=xrangemin   # 十进制数的最小值范围
        self.xrangemax=xrangemax   #十进制数的最大值
        self.crossrate=1           #交叉率
        self.mutationrate=0.01     #变异率


    #初始化种群
    def initalpop(self):
        #初始化popsize个向量 每个向量的长度都为chrosize
        pop=random.randint(0,2,size=(self.popsize,self.chrosize))
        return pop

    '''
    将二进制转化到求解空间的数值  即将二进制数据转换成十进制
    '''
    def get_declist(self,chroms):
        #精度
        step=(self.xrangemax-self.xrangemin)/float(2**self.chrosize-1)
        self.chroms_declist=[]
        for i in range(self.popsize):
            chrom_dec=self.xrangemin+step*self.chromtodec(chroms[i])
            self.chroms_declist.append(chrom_dec)
        #print(self.chroms_declist)
        return self.chroms_declist

    '''
    将二进制数组转换成十进制 参数为：染色体（即编码后的个体）
    '''
    def chromtodec(self,chrom):
        r=0
        m=1
        for i in range(self.chrosize):
            r=r+m*chrom[i]
            m=m*2  #越高位权重越大
        #返回这个十进制的数
        return r

    ''''
    原函数与适应度函数
    '''
    #对传入的每个变量 计算函数值
    def fun(self,x1):
        return x1*sin(10*pi*x1)+2.0
    #注意 这里的x是个向量 代表一条编码的染色体
    def get_fitness(self,x):
        fitness=[]
        for x1 in x:
            fitness.append(self.fun(x1))
        return fitness

    ''''
    轮盘赌算法 即各个个体被选中的概率与其适应度大小成正比 
    参数为：上一代的种群，上一代种群的适应度列表 返回值  挑选出来的个体
    '''
    def selection(self,popsel,fitvalue):
        new_fitvalue=[]
        #适应值之和
        totalfit=sum(fitvalue)
        #此变量存储累计概率
        accumulator=0.0
        for val in fitvalue:
            possible=(val*1.0/totalfit)
            accumulator+=possible
            new_fitvalue.append(accumulator)

        ms=[]
        #随机生成popsze个随机树 相当于随机抽取popsize次
        for i in range(self.popsize):
            #随机生成0-1之间的随机数
            ms.append(random.random())

        #对生成的随机数据进行排序
        ms.sort()
        fitin=0
        newin=0
        #将newpop引用指向上一代种群 （注意这里是数组 所以是引用的关系）
        newpop=popsel

        ####要理解这几行代码要举个实际的例子看
        while newin <self.popsize:
            #随机投掷 选择落入个体所占轮盘空间的个体
            if(ms[newin]<new_fitvalue[fitin]):
                #fitin指代的当前染色体被选中
                newpop[newin]=popsel[fitin]
                newin=newin+1
            else:
                #fitin 为指向fitvalue数组的索引
                fitin=fitin+1
        #适应度大的个体会被选择的概率较大 使得新种群中 会有重复的较优个体
        pop=newpop

        return pop

    '''
    染色体交叉 参数为：种群所有个体  返回经过重组后的所有个体
    重组的规则为相邻的两个个体进行单点交叉 返回重组后的种群所有个体
    '''
    def crossover(self,pop):
        for i in range(self.popsize-1):
            #近邻个体交叉 若随机数小于交叉率
            if(random.random()<self.crossrate):
                #随机选择交叉点
                singpoint=random.randint(0,self.chrosize)
                temp1=[]
                temp2=[]

                #对个体进行切片 重组
                temp1.extend(pop[i][0:singpoint])
                temp1.extend(pop[i+1][singpoint:self.chrosize])
                temp2.extend(pop[i + 1][0:singpoint])
                temp2.extend(pop[i][singpoint:self.chrosize])
                pop[i]=temp1
                pop[i+1]=temp2
        return pop

    '''
    变异 参数：变异前的种群 返回值：变异后的种群
    '''
    def mutation(self,pop):
        for i in range(self.popsize):
            #翻转变异 随机数小于变异率 进行变异
            if (random.rand()<self.mutationrate):
                #挑选一个随机点
                mpoint=random.randint(0,self.chrosize-1)
                #将随机点上的基因进行反转
                if(pop[i][mpoint]==1):
                    pop[i][mpoint]=0
                else:
                    pop[i][mpoint]=1
        return pop

    '''
    精英保留策略
    通过选择、交叉和变异创造新个体的同时，可能会失去一部分优秀的个体，
    总体思想是：判断经过选择、交叉、变异之后的种群中，是否产生了更优秀的个体，如果没有，则将上一代的精英个体
    替换较差的个体 可以防止优秀个体的丢失
    # 传入参数：
    当前种群，上一代最优个体，选择交叉变异后的最优适应值，上一代的的最优适应度
    '''
    def elitism(self,pop,popbest,nextbestfit,fitbest):
        #这些变量都是在主函数中生成的
        if nextbestfit-fitbest<0:
            #满足精英策略之后，找到最差个体的索引 进行替换
            pop_worst=nextfitvalue.index(min(nextfitvalue))
            pop[pop_worst]=popbest
        return pop




if __name__ == '__main__':
    #遗传代数
    generation=100
    #引入Gas类，传入参数 ：种群大小、编码长度、变量范围
    mainGas=Gas(100,10,-1,2)
    pop=mainGas.initalpop() #种群初始化
    #每代的最高适应度
    pop_best=[]
    for i in range(generation):
        #在遗传代数内进行迭代
        declist=mainGas.get_declist(pop) #解码
        #fitvalue中保存着每个个体的适应度
        fitvalue=mainGas.get_fitness(declist) #适应度

        #选择适应度函数最高的个体
        popbest=pop[fitvalue.index(max(fitvalue))]
        #对popbest进行深复制 目的是为了后面的精英选择做准备
        popbest=copy.deepcopy(popbest)
        #最高适应度
        fitbest=max(fitvalue)
        #保存每代的最高适应度
        print('fitbest:',fitbest)
        pop_best.append(fitbest)


        #####################进行算子运算 并不断更新更新pop##############################3

        mainGas.selection(pop,fitvalue) #选择
        mainGas.crossover(pop)          #交叉
        mainGas.mutation(pop)           #变异

        ######################精英策略前的准备##########################################3
        #对变异之后的pop 求解最大适应度
        nextdeclist=mainGas.get_declist(pop)
        nextfitvalue=mainGas.get_fitness(nextdeclist)
        nextbestfit=max(nextfitvalue)

        ##############################精英策略###########################################
        #比较深复制的个体适应度和变异之后的适应度
        # 传入参数：当前种群，上一代最优个体，选择交叉变异后的最优适应值，上一代的的最优适应度
        mainGas.elitism(pop,popbest,nextbestfit,fitbest)


    #对结果进行可视化分析
    t=[x for x in range(generation)]
    #pop_best中保存着历代的最优的适应值
    s=pop_best

    plt.plot(t,s)
    plt.show()
    plt.close()










