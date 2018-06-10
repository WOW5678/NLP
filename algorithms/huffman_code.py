# -*- coding:utf-8 -*-
#功能：

#six是用在byte和int类型之间的的转换，在压缩和解压缩文件中会使用到
import six
#sys包是用来获取main函数当中的参数
import sys


'''
基类 HuffNode
'''
class HuffNode(object):
    '''
    定义一个HuffNode虚类 包含两个虚方法：
    （1）获取节点的额权重函数
    （2）获取此节点是否为叶子节点
    '''
    def get_weight(self):
        raise NotImplementedError("The Abstract Node Class doesn't define 'get_weight'")
    def isleaf(self):
        raise NotImplementedError("The Abstract Node Class doesn't define 'isleaf'")

'''
派生类 LeadNode
'''
class LeafNode(HuffNode):
    '''
    叶子结点类
    '''
    def __init__(self,value=0,freq=0):
        '''
        :param value: 要编码的字符
        :param freq:  要编码字符的权重
        :return: 
        '''
        super(LeafNode,self).__init__()
        self.value=value
        self.weight=freq

    def isleaf(self):
        '''
        必须要实现的基类的方法
        '''
        return True
    def get_weight(self):
        """
        必须要实现的基类方法
        :return: 结点的权重
        """
        return self.weight

    def get_value(self):
        '''
        :return:叶子节点代表的字符的值 
        '''
        return self.value
'''
派生类 IntlNode 
'''
class IntlNode(HuffNode):
    '''
    中间结点类
    '''
    def __init__(self,left_child=None,right_child=None):
        '''
        :param left_child: 左结点
        :param right_child: 右结点
        '''
        super(IntlNode,self).__init__()
        #节点的值 为左节点的权重+右节点的额权重之和
        self.weight=left_child.get_weight()+right_child.get_weight()
        #左右结点
        self.left_child=left_child
        self.right_child=right_child

    def isleaf(self):
        '''
        必须要实现的基类方法
        '''
        return False
    def get_weight(self):
        """
        必须要实现的基类方法
        """
        return self.weight
    def get_left(self):
        '''
        获取左孩子结点
        '''
        return self.left_child
    def get_right(self):
        """
        获取右孩子结点
        """
        return self.right_child

'''
类 HuffTree
'''
class HuffTree(object):
    '''
    哈夫曼树
    '''
    def __init__(self,flag,value=0,freq=0,left_tree=None,right_tree=None):
        """
        :param flag: 标记这棵树是否只含一个结点
        :param value: 如果只含一个结点 value为叶节点表示的字符
        :param freq: 如果只含一个结点 freq为叶节点的权重
        :param left_tree: 左子树
        :param right_tree: 右子树
        """
        if flag==0:
            self.root=LeafNode(value,freq)
        else:
            self.root=IntlNode(left_tree.get_root(),right_tree.get_root())


    def get_root(self):
        '''
        获取树的根结点
        '''
        return self.root
    def get_weight(self):
        ''''
        获取这个树的根结点的权重
        '''
        return self.root.get_weight()

    def traverse_huffman_tree(self,root,code,char_freq):
        '''
        利用递归方法遍历huffman_tree，并且以此方法得到每个字符对应的huffman编码保存到
        字典char_freq中
        :param root: 
        :param code: 
        :param char_freq: 
        :return: 
        '''
        if root.isleaf():
            char_freq[root.get_value()]=code
            #print('root.get_value:',type(root.get_value()))
            asc=six.byte2int(root.get_value())

            print("it=%d, char=%c, freq=%d, code=%s" %(asc,chr(asc),root.get_weight(),code))
        else:
            self.traverse_huffman_tree(root.get_left(),code+'0',char_freq)
            self.traverse_huffman_tree(root.get_right(),code+'1',char_freq)

def buildHuffmanTree(list_hufftrees):
    '''
    构造huffman tree
    :param list_hufftree: 
    :return: 
    '''
    while len(list_hufftrees)>1:
        #1.按照weight 对huffman从小到大排序
        list_hufftrees.sort(key=lambda x:x.get_weight())

        #2.挑选出weight最小的两个huffman编码树
        temp1=list_hufftrees[0]
        temp2=list_hufftrees[1]
        list_hufftrees=list_hufftrees[2:]

        #3.构造一个新的huffman树
        newed_hufftree=HuffTree(1,0,0,temp1,temp2)

        #4.放入到数组当中
        list_hufftrees.append(newed_hufftree)

    #最后，数组中最后剩下的那棵树就是构造的huffman编码树
    return list_hufftrees[0]

if __name__ == '__main__':
    # #获取用户的输入
    # if len(sys.argv)!=2:
    #     print('please input inputfilename')
    #     exit(0)
    # else:
    #     INPUTFILE=sys.argv[1]
    INPUTFILE='data1.txt'
    #1. 以二进制的方式打开文件
    f=open(INPUTFILE,'rb')
    filedata=f.read()
    #获取文件的字节总数
    filesize=f.tell()
    print(filesize)

    #2.统计byte的取值[0-255]的每个值出现的频率
    char_freq={}
    for x in range(filesize):
        if filedata[x]=='\n' or filedata[x]=='\r':
            continue
        #将字节转换成int型的数据
        tem=six.int2byte(filedata[x])
        if tem in char_freq.keys():
            char_freq[tem]=char_freq[tem]+1
        else:
            char_freq[tem]=1

    #输出统计结果
    # for tem in char_freq.keys():
    #     print (tem,":",char_freq[tem])

    #3. 开始构造原始的huffman编码数组，用于构造huffman编码树
    list_hufftrees=[]
    for x in char_freq.keys():
        #使用huffmanTree的构造函数 定义一颗只包含一个叶子节点的huffman树
        print(x,':',char_freq[x])
        tem=HuffTree(0,x,char_freq[x],None,None)
        #将其添加到list_hufftrees中
        list_hufftrees.append(tem)
    #print ('list_hufftrees:',list_hufftrees)

    #5. 构造huffman编码树，并且获取到每个字符对应的编码 并打印出来
    tem=buildHuffmanTree(list_hufftrees)
    tem.traverse_huffman_tree(tem.get_root(),' ',char_freq)


