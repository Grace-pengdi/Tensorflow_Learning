import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

def  add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#create data
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise

#use placeholder to define the inputs
#None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#define nueral layers
#hidden layer,input onle 1 feature
layer1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#output layer,use layer1 as input
prediction=add_layer(layer1,10,1,activation_function=None)

#loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))  #reduction_indices代表要被reduce的那个维度

#train algorithm(method)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#plot the real data
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()  #用于连续显示
plt.show()

#start train
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value=sess.run(prediction,feed_dict={xs:x_data})

            #plot the prediction line, line width=5,color=red
            lines=ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1) #暂停0.1s
            