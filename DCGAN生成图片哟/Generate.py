#coding:UTF-8
"""
用DCGAN的生成器模型和训练的到的生成器参数文件来生成图片
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from network import *


def generate():
    #构造生成器
    g=generate_model()
    #配置生成器
    g.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(LEARNING_RATE,beta_1=BETA_1))

    #加载训练好的生成器的参数
    g.load_weights("./model/generator_weight/")

    #连续性均匀分布随机数据
    random_data=np.random.uniform(-1,1,(BATCH_SIZE,100))

    #用随机数据作为输入生成器，生成图片数据
    images=g.predict(random_data,verbose=1)

    #用生成的图片数据生成PNG图片
    for i in range(BATCH_SIZE):
        image=images[i]*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save("./images/output/output3/image-%s.png"%i)



if __name__=="__main__":
    generate()
