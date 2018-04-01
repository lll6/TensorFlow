import  tensorflow as tf

input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.constant([2.0,3.00,4.0],name='input2')
output = tf.add_n([input1,input2],name = "add")

with tf.Session() as sess:
    writer = tf.summary.FileWriter("F:/TF/graphs",sess.graph)
    print(sess.run(output))

writer.close()