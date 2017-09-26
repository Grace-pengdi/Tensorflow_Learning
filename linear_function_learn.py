import tensorflow as tf 
import numpy as np 

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#Construct a model
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

#loss function
loss=tf.reduce_mean(tf.square(y-y_data))

#training algorithm
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)  #update args

#start training
init=tf.global_variables_initializer()

#creat a session to run every training data 
sess=tf.Session()  #where operation are excuted
sess.run(init)   #Very important step

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))