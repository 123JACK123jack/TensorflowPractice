#coding：UTF-8
"""
DCGAN深层卷积的生成对抗网络
"""
import  tensorflow as tf

#超参数：
EPOCHS=100#轮数
BATCH_SIZE=50
LEARNING_RATE=0.0002
BETA_1=0.5

#定义判别器的模型

def discriminator_model():
    model=tf.keras.Sequential()
#添加一个卷积层
    model.add(tf.keras.layers.Conv2D(
        64,#64个卷积核，输出的深度
        (5,5),#卷积核大小
        padding='same',#补零  输出的大小不变  补零两圈
        input_shape=(64,64,3)#输入形状，64，64，3 表示三通道，彩色RGB
    ))
#添加一个激活层 tanh
    model.add(tf.keras.layers.Activation(
        "tanh"
    ))

#添加一个池化层
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2,2)
    ))
#t添加一个卷积层
    model.add(tf.keras.layers.Conv2D(
        128,
        (5,5)
    ))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    ))
    model.add(tf.keras.layers.Conv2D(
        128,
        (5, 5)
    ))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    ))
    model.add(tf.keras.layers.Flatten())#扁平化
    model.add(tf.keras.layers.Dense(1024))#1024个神经元的全连接层
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))#添加sigmoid激活函数

    return  model


#定义生成器模型
#从随机数来生成图片
def generate_model():
    model=tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(input_dim=100,units=1024))#input_dim输入的维度
    #输入维度100，输出维度为1024的全连接层
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128*8*8))#8192个神经元的全连接层
    model.add(tf.keras.layers.BatchNormalization())#对数据进行批标准化
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Reshape((8,8,128),input_shape=(128*8*8,)))#8乘8像素
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))#相当于pooling的反操作   变成16×16
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))#变成了32×32
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))#变成了64×64
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same"))#深度要变成图片的深度3
    model.add(tf.keras.layers.Activation("tanh"))

    return model

#构造一个 Sequential对象，包含生成器和判别器
#输入：经过生成器流到判别器，输出一个判断
def generator_containing_discriminator(generator,discriminator):
    model=tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable=False#初始时判别器是不可被训练
    model.add(discriminator)
    return model





















