#coding:utf-8

#用训练好的神经网络参数来作曲

import  numpy as np
from utils1 import *
from Music import *
import tensorflow as tf

#以之前训练得到的最佳参数来生成音乐
def generate():
    #加载用于训练神经网络的音乐数据
    with open('./Data/data1/1.txt','rb') as filepath:
       notes=pickle.load(filepath)

   #得到所有音调的名字
    pitch_names=sorted(set(item for item in notes))
    #得到所有不重复的音调数目
    num_pitch=len(set(notes))

    normalized_input,network_input=prepare_sequences(notes,pitch_names,num_pitch)

    #载入之前训练得到的最好的参数 来生成神经网络模型
    model=generateNetwork(normalized_input,num_pitch,weights_file="./Data/data1/Weight-02-4.5669.hdf5")

    #用神经网络来生成音乐数据
    prediction=generate_notes(model,network_input,pitch_names,num_pitch)

    #将预测的音乐数据生成MIDI文件，转换成MP3文件
    create_music(prediction)

def prepare_sequences(notes,pitch_names,num_pitch):
    #为神经网络准备好供训练的序列
   sequence_length=100#序列的长度

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
   normalized_input=np.reshape(network_input,(n_patterns,sequence_length,1))
   #将输入做标准化（归一化）
   normalized_input=normalized_input/float(num_pitch)
   #将期望输出转换成{0，1}组成的布尔矩阵     #########因为计算损失是用交叉熵的方式用类似于布尔矩阵的方式
   network_output=tf.keras.utils.to_categorical(network_output)


   return(normalized_input,network_input)

def generate_notes(model,network_input,pitch_names,num_pitch):
    """
    基于一序列音符用神经网络来生成新的音符
    """
    #从输入里随机选择一个序列作为预测生成的音乐的起始点
    start=np.random.randint(0,len(network_input)-1)

    #创建一个字典用于映射整数和音调
    int_to_pitch=dict((num,pitch)for num ,pitch in enumerate(pitch_names))

    pattern=network_input[start]

    #神经网络实际生成的音调
    prediction_output=[]
    #生成700个音调/音符
    for note_index  in range(500):
        prediction_input=np.reshape(pattern,(1,len(pattern),1))
        #预测输入的归一化处理
        prediction_input=prediction_input/float(num_pitch)

        #用载入来训练所的最佳参数文件的神经网路来  预测/生成新的音调/音符
        prediction=model.predict(prediction_input,verbose=0)

        #argmax取predition最大的维度类似onehot编码
        index=np.argmax(prediction)

        #将整数值转成对应的音调
        result=int_to_pitch[index]

        prediction_output.append(result)

        #往后移动pattern
        pattern.append(index)
        pattern=pattern[1:len(pattern)]

    return prediction_output




if __name__=='__main__':
     generate()


