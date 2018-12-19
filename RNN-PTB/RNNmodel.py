#RNN-LSTM
import tensorflow as tf
#神经网络模型
class Model(object):
    def __init__(self,input,is_training,hidden_size,vocab_size,num_layers,dropout=0.5,
                 init_scale=0.05):
        self.is_training=is_training
        self.input_obj=input
        self.batch_size=input.batch_size
        self.num_steps=input.num_steps
        self.hidden_size=hidden_size
        #在这里的操作和变量用CPU来实现
       # with tf.device("/cpu:0"):
            #创建词向量，词向量本质上是一种单词聚类的方法
        embedding=tf.Variable(tf.random_uniform([vocab_size,self.hidden_size],-init_scale,init_scale))#表示映射
        inputs=tf.nn.embedding_lookup(embedding,self.input_obj.input_data)#查询

        if is_training and dropout<1:
                inputs=tf.nn.dropout(inputs,dropout)
            #状态的存储和提取
            #第二维是因为对每一个LSTM单元有两个来自上一单元的输入
            #一个是前一时刻LSTM的输出h(t-1)
            #一个是前一时刻的单元状态C（t-1）
            #这个C和j是用于构建之后的tf.contrib.rnn.LSTMStateTuple
        self.init_state=tf.placeholder(tf.float32,[num_layers,2,self.batch_size,self.hidden_size])

            #每一层的状态
        state_per_layer_list=tf.unstack(self.init_state,axis=0)
            #初始的状态   包含前一时刻LSTM的输出和前一时刻单元状态  用于之后的dynamic_rnn
        rnn_tuple_state=tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
                 for idx in range(num_layers)] #0和1分别对应c和h
            )
            #创建一个LSTM层，其中的神经元数目就是hidden_size个，默认650
        cell=tf.contrib.rnn.LSTMCell(hidden_size)

        if is_training and dropout<1:
                cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=dropout)

            #如果LSTM的层数大于1，则总计创建num_layers个LSTM层
            #并将所有的LSTM层包装进MultiRNNCell这样的序列化层级模型中
            #state_is_tuple=True表示接受LSTMStateTuple形式的输入状态
        if num_layers>1:
                cell=tf.contrib.rnn.MultiRNNCell([cell  for _ in range(num_layers)],state_is_tuple=True)
           #dynamic_rnn动态RNN，可以让不同迭代传入的batch可以是长度不同的数据
        #但是同一次迭代中一个Batch内部的所有数据和长度仍然是固定的
        #dynamic_rnn能更好的处理padding（补0）的情况，节约计算资源
        #返回两个变量:
        #第一个是一个Batch里在时间维度默认是35上展开的所有LSTM单元的输出，形状默认是【20，35，650】之后会经过扁平处理
        #第二个是最终i的state状态，包含当前时刻LSTM的输出h（t）和当前时刻的单元状态C（t)
        output,self.state=tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32,initial_state=rnn_tuple_state)
   #扁平化处理，改变输出形状为(batch_size*num_step,hidden_size),形状默认为【700，650】
        output=tf.reshape(output,[-1,hidden_size])
        #softmax的权重
        softmax_w=tf.Variable(tf.random_uniform([hidden_size,vocab_size],-init_scale,init_scale))
        softmax_b=tf.Variable(tf.random_uniform([vocab_size],-init_scale,init_scale))
#logits是逻辑回归模型（用于分类），计算的结果
#这个logits之后会用softmax来转层百分比概率
#output是输入x，softmax_w是权重，softmax_b是偏置
#返回wx+b的值
        logits=tf.nn.xw_plus_b(output,softmax_w,softmax_b)
        #将logits转化为三维的tensor为了sequence loss的计算
        #形状默认为【20，35，10000】
        logits=tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])
        #计算logits的序列交叉熵 Cross-Entory的损失
        loss=tf.contrib.seq2seq.sequence_loss(
            logits,#形状默认为20，35，10000
            self.input_obj.targets,#目标期望输出形状为【20，35】
            tf.ones([self.batch_size,self.num_steps],dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True

        )
        #更新代价
        self.cost=tf.reduce_sum(loss)

        self.softmax_out=tf.nn.softmax(tf.reshape(logits,[-1,vocab_size]))#【700，10000】

        #取最大概率的值作为预测
        self.predict=tf.cast(tf.argmax(self.softmax_out,axis=1),tf.int32)#转成什么类型  10000的纬度上

  #预测值和真实值的对比
        correct_prediction=tf.equal(self.predict,tf.reshape(self.input_obj.targets,[-1]))#equal是否相等
        #计算预测的进度
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#tf.cast计算精度
        if  not is_training:
            return
        #学习率
        self.learning_rate=tf.Variable(0.0,trainable=False)
        #返回所有可被训练
        #也就是除了不可被训练的学习率之外的其他变量
        tvars=tf.trainable_variables()


        #tf.clip_by_global_normal(实现梯度裁剪，是为了防止梯度爆炸

        #tf.gradients计算self.cost对于tvars的梯度 求导，返回一个梯度列表
        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,tvars
                                                    ),5)
        #优化器用GradientDesecentOptimizer梯度优化器
        optimizer=tf.train.GradientDescentOptimizer(self.learning_rate)

        #apply_gradients  将之前用Gradient Cliping梯度裁剪过的梯度应用到可被训练的变量上去，做梯度下降
        #apply_gradient其实是minimize方法里面的第二步，第一步是计算梯度  然后再修剪一下，其实和minilize是一样的
        self.train_op=optimizer.apply_gradients(zip(grads,tvars),
                                                global_step=tf.train.get_or_create_global_step())
        #更新学习率
        self.new_lr=tf.placeholder(tf.float32,shape=[])
        self.lr_update=tf.assign(self.learning_rate,self.new_lr)#assign赋值操作
    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})
