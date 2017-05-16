
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv

with open('dataset1_65_10_3000.csv', 'r') as f:     #Prev traintestEnglishFinal.csv
  reader = csv.reader(f)
  your_list = list(reader)

b = np.array(your_list)
#np.set_printoptions(threshold='nan')
#print(b)

#classmap = {'A': 1, 'B': 2, 'C':3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10}
classmap = {'A': 0, 'B': 1, 'C':2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
featureArray = [b[i][0:65] for i in range(0,3000)]     #Prev [b[i][0:3024] for i in range(0,4500)]

classArray = [b[i][65:66] for i in range(0,3000)]   #Prev [b[i][3024:3025] for i in range(0,4500)]

ftrain = np.array(featureArray)

ctrain = np.array(classArray)


ti  = []
for i in ftrain:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

print(train_input)


train_output = []
 
for i in ctrain:
    count = 0
    for j in i:
     count = int(j)
    temp_list = ([0]*10)
    temp_list[count]=1
    train_output.append(temp_list)
	
print(train_output)

#One-hot vector obtained

NUM_EXAMPLES = 2000 # Prev 3000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 2000
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 2,000

data = tf.placeholder(tf.float32, [None, 65,1])   #Prev 3024
target = tf.placeholder(tf.float32, [None, 10])  # Prev [None, 15]

num_hidden = 60 #Changed from 128
cell = rnn.BasicLSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))  #changed

global_step = tf.Variable(0, trainable = False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           50, 0.50, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate,0.9)
minimize = optimizer.minimize(cross_entropy,global_step=global_step)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
 
batch_size = 60 #Changed from 200
no_of_batches = int(len(train_input)/batch_size)
epoch = 1000   # Prev 100
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    incorrect = sess.run(error,{data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
