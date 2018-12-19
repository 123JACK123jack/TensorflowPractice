#coding:utf-8

#训练神经网络，将参数Weighy存入HDF5文件

import  numpy as np
from utils1 import *
from Music import *
import tensorflow as tf

def train():
    notes=get_notes()
    #得到所有不重复的音调数
    num_pitch=len(set(notes))
    network_input,network_output=prepare_sequences(notes,num_pitch)
    model=generateNetwork(network_input,num_pitch)

    filepath="./Data/data1/Weight-{epoch:02d}-{loss:.4f}.hdf5"
    #用checkpoint（检查点）文件在每一轮epoch结束时保存模型参数，不怕训练过程
    #中丢失模型参数，可以在我们对loss损失满意的时候随时去停止训练

    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',#监视器，衡量哪个值
        verbose=0,
        save_best_only=True,#较好结果不会覆盖其他
        mode="min"#模式：让loss尽量小，所以关注它的越小越好

    )
    callbacks_list=[checkpoint]
    #y用fit来训练模型
    model.fit(network_input,network_output,epochs=2,batch_size=60,callbacks=callbacks_list)

def prepare_sequences(notes,num_pitch):
    #为神经网络准备好供训练的序列
   sequence_length=50#序列的长度
   #得到所有不同音调的名字
   pitch_names=sorted(set(item for item in notes))

   #把音调转成整数，创建一个字典 用于映射 音调 和 整数

   pitch_to_int=dict((pitch,num) for num,pitch in enumerate(pitch_names))#enumerate来生成每个音调对应的数字
   network_input=[]
   network_output=[]#实际位置上的实际值

   for i in range(0,len(notes)-sequence_length,1):
            sequence_in=notes[i:i+sequence_length]
            sequence_out=notes[i+sequence_length]

            network_input.append([pitch_to_int[char]for char in sequence_in])
            network_output.append([pitch_to_int[sequence_out]])#相当于标签值

   n_patterns=len(network_input)
   #将输入序列的形状转成 神经网络模型可以接受的
   network_input=np.reshape(network_input,(n_patterns,sequence_length,1))
   #将输入做标准化（归一化）
   network_input=network_input/float(num_pitch)
   #将期望输出转换成{0，1}组成的布尔矩阵     #########因为计算损失是用交叉熵的方式用类似于布尔矩阵的方式
   network_output=tf.keras.utils.to_categorical(network_output)


   return(network_input,network_output)

if __name__=='__main__':
     train()


