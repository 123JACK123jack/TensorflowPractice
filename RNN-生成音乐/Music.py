#RNN循环神经网络模型
import   tensorflow as tf
def generateNetwork(inputs,num_pitch,weights_file=None):
    model=tf.keras.Sequential()#定义一个模子
    #添加第一层LSTM
    model.add(tf.keras.layers.LSTM(512,input_shape=(inputs.shape[1],inputs.shape[2]),return_sequences=True)) #LSTM层是神经元的数目是512，也是LSTM输出的维度
#对第一个LSTM层必须要设置，   retyrn_sequende默认是false ，表示只输出最后一个输出给下一层，对于堆叠的LSTM，第一第二层除了最后一层全都需要将所有结果输出到下一层
#最后一层LSTM则默认即可，只输出最后一个输出结果给其他神经网络层
    #丢弃百分之三十   防止过拟合
    model.add(tf.keras.layers.Dropout(0.3))
    #第二层LSTM
    model.add(tf.keras.layers.LSTM(512,return_sequences=True))
    #丢弃百分之三十
    model.add(tf.keras.layers.Dropout(0.3))
    #第三层LSTM
    model.add(tf.keras.layers.LSTM(512))#return_sequence默认False最后一层LSTM只返回输出序列最后一个
    #全连接层  256个神经元
    model.add(tf.keras.layers.Dense(256))
    #丢弃百分之三十
    model.add(tf.keras.layers.Dropout(0.3))
    #全连接层  神经元数目是音调的数目
    model.add(tf.keras.layers.Dense(num_pitch))#输出的数目等于所有不重复的音调的数目
    model.add(tf.keras.layers.Activation('softmax'))#用softmax激活函数计算生成每个音符对应的概率
    #制定一些参数   交叉熵计算误差  优化器是rmsprop 对循环神经网络来说比较优秀的（RMSProp)
    model.compile(loss="categorical_crossentropy",optimizer="rmsprop")

    if weights_file is not None:#如果是生成音乐时
        #从 HDF5  文件中加载所有神经网络层的参数（Weighs
        model.load_weights(weights_file)


    return  model


if __name__=='__main__':
    generateNetwork()
