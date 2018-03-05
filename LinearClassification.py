import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

mem = Memory("./mycache")
#X =np.zeros([690,14],dtype=float)
#print(X)
@mem.cache
def get_data():
    data = load_svmlight_file('F:/MLworkplace/australian_scale')
    return data[0], data[1]

X, Y = get_data()
Y = Y.reshape((690,1))
X = X.toarray()
print(type(X))
print(type(Y))
print(X.shape)
batch_size = 8

#variable
w1 = tf.Variable(tf.random_normal([14,5],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([5,1],stddev=1,seed=1))

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
#feedin
x_data = tf.placeholder(tf.float32,shape=[None,14],name = 'x1-input')
y_taget = tf.placeholder(tf.float32,shape=[None,1],name = 'x2-input')

a = tf.matmul(x_data,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(
    y_taget * tf.log(tf.clip_by_value(y,1e-10,1.0))
)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#data
# rdm = RandomState(1)
# dataset_size = 128
# X = rdm.rand(dataset_size,2)
# Y = rdm.rand(dataset_size,1)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#initialize_all_variables()
    sess.run(init_op)
    sess.run(w1)
    sess.run(w2)

    steps = 5000
    for i in range(steps):
        start = (i*batch_size)%690
        end = min(start+batch_size,690)

        sess.run(train_step,feed_dict={x_data: X[0:8],y_taget: Y[0:8]})

        if i % 1000 ==0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x_data:X,y_taget:Y})
            print(i,total_cross_entropy)
