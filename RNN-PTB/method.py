# #RNN生成和预测句子
# #用到的数据集：PTB数据集
# 结果是当前状态和以前记忆的叠加
# RNN的优势：
# 每一个输出与前面的输出建立起关联
# 能够很好地处理序列化数据 音乐文章
# 能以前面的序列化对象为基础，来生成新的序列化对象
# 方向传播是对loss函数求导 梯度进行反向穿播
# 容易产生梯度消失和梯度爆炸
# 梯度裁剪来解决梯度爆炸
# 梯度消失类似于记忆消散
# LSTM来解决
# 长的短期记忆  处理的记忆都是短期的记忆，所以LSTM维持这些短期记忆
# 多了一个单元状态C来保持长期状态
# 三个输入：Xt当前输入   前一时刻LSTm的输出  ，前一时刻的单元状态的输出
# 两个输出：当前时刻lstm输出，当前单元状态的输出
# LSTM的三重门机制：Ct-1前一时刻的长期状态， Ct当前时刻的长期状态
# ht当前时刻的输出    C‘t当前时刻的即时状态
# 遗忘门，完全遗忘前一时刻的长期状态，0-1
# 输入们：0-1 控制输入
# 输出门：0-1
# 0完全舍弃 1完全开放
# 重要性：遗忘门，输入们，输出门
# 解决原理：
# GRU  LSTM 的变体
# WordEmbedding：词向量/嵌入
# one-hot编码模式，数据量很大时效率很低
# # 词向量 数据线比较大的时候
# 单词相关性比较大的时候距离比较近
# one-hot编码与词向量编码的转换

#
import  os
import sys
import argparse#参数解析
import datetime
import collections
import numpy as np
import  tensorflow as tf

data_path="./Data/data"
#保存训练所的模型参数文件的目录
save_path='./Save'
#测试时读取模型参数文件的名称：
load_file="train-checkpoint-69"
parser=argparse.ArgumentParser()#参数解析器
parser.add_argument('--data_path',type=str,default=data_path,help='数据路径')
parser.add_argument('--load_file',type=str,default=load_file,help='the path of checkpoint')
args=parser.parse_args()#实际去解析参数
Py3=sys.version_info[0]==3

#将文件根据语句结束标识符（<eos>来分割
def read_words (filename):
    with open(filename,"r")  as f:
        if Py3:
            return f.read().replace("\n","<eos>").split()
#构造从单词到唯一整数值的映射
def build_vocab(filename):
    data=read_words(filename)

    counter=collections.Counter(data)#统计单词出现的频数
    count_pairs=sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    words,_=list(zip(*count_pairs))#按照顺序从0-9999所以出现次数最多的单词对应数字0
    #单词到整数的映射
    word_to_id=dict(zip(words,range(len(words))))

    return word_to_id
#将文件里的单词都替换成独一的整数
def file_to_id_word_ids(filename,word_to_id):
    data=read_words(filename)
    return[word_to_id[word] for word in data if word in word_to_id]

#加载所有数据，读取所有单词，吧其转化为唯一对应的整数值
def load_data(data_path):
    train_path=os.path.join(data_path,"ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    #建立词汇表，将所有单词转为唯一对应的整数
    word_to_id=build_vocab(train_path)

    train_data=file_to_id_word_ids(train_path,word_to_id)
    valid_data = file_to_id_word_ids(valid_path, word_to_id)
    test_data = file_to_id_word_ids(test_path, word_to_id)

    #s所有独一的词汇的个数
    vocab_size=len(word_to_id)
    #反转一个词汇表:为了之后从整数转为单词
    id_to_word=dict(zip(word_to_id.values(),word_to_id.keys()))#word_to_id键是单词，值是对应的整数，所以交换一下，键值为整数，值是单词
    print(train_data[:10])
    print(word_to_id)
    print(vocab_size)
    return train_data,valid_data,test_data,vocab_size,id_to_word
#生成批次样本
def generate_batches(raw_data,batch_size,num_steps):
    #将数据转为Tensor的类型
    raw_data=tf.convert_to_tensor(raw_data,name="raw_data",dtype=tf.int32)
    data_len=tf.size(raw_data)
    batch_len=data_len//batch_size
    #将数据转为[batch_size,batch_len]
    data=tf.reshape(raw_data[0:batch_size*batch_len],[batch_size,batch_len])
    epoch_size=(batch_len-1)//num_steps

    #range_input_producer可以用多线程异步的方式从数据集里提取数据
    #用多线程可以加快训练，因为feed_dict的赋值方式效率不高
    #shuffle为False表示不大乱数据而按队列先进先出的方式提取数据
    i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
    #假设一句话是：我爱我的祖国和人民
    #如果x是类似这样：我爱我的祖国
    x=data[:,i*num_steps:(i+1)*num_steps]
    x.set_shape([batch_size,num_steps])
    y=data[:,i*num_steps+1:(i+1)*num_steps+1]
    y.set_shape([batch_size,num_steps])
    return x,y#实际的x和y
#输入数据
class Input(object):
    def __init__(self,batch_size,num_steps,data):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.epoch_size=((len(data)//batch_size-1)//num_steps)
        self.input_data,self.targets=generate_batches(data,batch_size,num_steps)
if __name__=="__main__":
    load_data()

#batch_zize一次迭代所用的样本数目，越大，所需的内存越大
#Eopch所有训练样本完成一次  纪元
# 超参数：
# init_scale 权重参数的初始取值跨度，一开始取小一些有利于训练
# learn_rate 学习率，训练时初始为1.0
# num_layers , LSTM层的数目，默认是2
# num_steps  LSTM展开的步数，相当于每个批次输入单词的数目 默认是35
# hidden_size  LSTM层的神经元数目，也是词向量的维度 默认是650
# max_lr_eopch  用初始学习率训练的Epoch数目，默认是10
# dropout  在Droupout层的留存率   默认是0.5
# lr_decay  在过了max_lr_epoch之后每一个EPOCH的学习率衰减率
# （20   batchsize ，30，650
# 取比较小的batchsize有利于随机梯度下降，防止困在局部最小值


