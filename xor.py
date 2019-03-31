# -*- coding: utf-8 -*- 
print("--------------------------start---------------------")

import tensorflow as tf

#生成训练集
X=[[0,0],
   [0,1],
   [1,0],
   [1,1]]
Y=[[0],[1],[1],[0]]

#定义并初始化神经网络的参数值
#值得注意的是！！隐层节点数至少为3！！
w1=tf.Variable(tf.random_normal([2,4],stddev=1,seed=1))
b1=tf.Variable(tf.zeros(4))

w2=tf.Variable(tf.random_normal([4,1],stddev=1,seed=1))
b2=tf.Variable(tf.zeros(1))
#训练集的输入格式
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#模拟神经网络的传输
a=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
y=tf.nn.sigmoid(tf.matmul(a,w2)+b2)

#定义输出损失函数和反向传播参数，比较核心的内容
loss=tf.reduce_mean(tf.multiply(y-Y,y-Y))
train_step=tf.train.AdadeltaOptimizer(0.9).minimize(loss)



with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    STEPS=10000
    for i in range(STEPS):
        sess.run(train_step,feed_dict={x:X,y_:Y})
        if i%10==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,loss is %g"%(i,total_loss))
    
    #进行预测
    print("predict:")
    yyy=sess.run(y,feed_dict={x:X})
    print(yyy)
    
    sess.close()

print("--------------------------end-----------------------")