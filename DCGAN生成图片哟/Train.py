#encoding:UTF-8


"""
训练DCGAN
"""

import glob
import numpy as np
from scipy import misc
import  tensorflow as tf
from network import *


def train():
    #获取训练数据
    data=[]
    for image in glob.glob("./images/react/*"):
        image_data=misc.imread(image)#imread利用PIL来读取图片数据
        data.append(image_data)


    input_data=np.array(data)
    print(input_data[1].shape)
    #将数据标准化为【-1，1】,这也是tanh激活函数的输出范围
    input_data=(input_data.astype(np.float32) - 127.5) / 127.5

    #构造生成器和判别器
    g=generate_model()
    d=discriminator_model()

    #g=构建生成器和判别其组成的网络模型
    d_on_g=generator_containing_discriminator(g,d)

    #优化器    用Adam Optimizer
    g_optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)


     #Sequential.compile
    #配置生成器和判别其
    g.compile(loss="binary_crossentropy",optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy",optimizer=g_optimizer)#用生成器的优化其
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)


    #开始训练
    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0]/BATCH_SIZE)):
            input_batch=input_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            #生成一个连续型均匀分布的随机数据（噪声）
            random_data=np.random.uniform(-1,1,size=(BATCH_SIZE,100))
            #生成器生成图片数据
            generate_images=g.predict(random_data,verbose=0)
           #连接你输入的图片数据和生成器生成的图片
            input_batch=np.concatenate((input_batch,generate_images))
            output_batch=[1]*BATCH_SIZE+[0]*BATCH_SIZE#相当于标签，前Batchsize是真实图片所以标签是1
            #后面的是你生成器生成的图片，不是真实的图片。所以便签是0

            #训练判别其   让它具备识别不合格生成图片的能力
            d_loss=d.train_on_batch(input_batch,output_batch)
            #当训练完判别器后 训练生成器时让判别器不可被训练
            d.trainable = False
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            #训练生成器
            g_loss=d_on_g.train_on_batch(random_data,[1]*BATCH_SIZE)

            #恢复判别器可被训练
            d.trainable=True
            #打印损失
            print("Step {} Generator Loss:{} Discriminator Loss:{}"
                  .format(index,g_loss,d_loss))

            #保存生成器和判别器的参数

            g.save_weights("./model/generator_weight/",True)
            d.save_weights("./model/discriminator_weight/",True)














if __name__=="__main__":
    train()

