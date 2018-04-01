#This ,py is for Logistic Regression with Softmax;

from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets(("/data/minist"),one_hot=True)

import tensorflow as tf

#Define parameter
learning_rate = 0.001
batch_size = 64
n_epochs = 30

#placeholder
X_data = tf.placeholder(tf.float32,[batch_size,784],name = "X_placeholder")
Y_target = tf.placeholder(tf.float32,[batch_size,10],name = "Y_placeholder")

#weights and bias
w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name = 'weights')
b = tf.Variable(tf.zeros([1,10],name = "bias"))

#model
logits = tf.matmul(X_data,w)+b

#loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_target,name='loss')
loss = tf.reduce_mean(entropy)

#define training
#using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    #start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n = 500
    print("start")
    for i in range(n):
        total_loss = 0
        for j in range(n):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            j,loss_batch = sess.run([optimizer,loss],feed_dict={X_data:X_batch,Y_target:Y_batch})
            total_loss  += loss_batch
        print('average loss epoch {0}:{1}'.format(i,total_loss/50))
    print('finished')

    total_correct_preds = 0
    for i in range(500):
        X_batch, Y_batch = MNIST.train.next_batch(batch_size)
        j, loss_batch, logits_batch = sess.run([optimizer, loss,logits], feed_dict={X_data: X_batch, Y_target: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)

    print('accuracy {0}'.format(total_correct_preds/MNIST.test.num_examples))