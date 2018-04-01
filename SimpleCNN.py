import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5

x0 =tf.placeholder(tf.float32,[None,n_inputs])
x1 =tf.placeholder(tf.float32,[None,n_inputs])

wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1,n_neurons],dtype=tf.float32))

y0 = tf.tanh(tf.matmul(x0,wx)+b)
y1 = tf.tanh(tf.matmul(y0,wy)+tf.matmul(x1,wx)+b)

x0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
x1_batch = np.array([[9,8,7],[6,5,4],[0,0,0],[3,2,1]])
print("start")
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    Y0_VAL,Y2_VAL = sess.run([y0,y1],feed_dict={x0: x0_batch,x1: x1_batch})
print(Y0_VAL)


