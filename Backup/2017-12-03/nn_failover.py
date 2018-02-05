import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()


# Read in Data and look at the first 100 datapoints
raw_data = pd.read_csv('log.csv')

data = raw_data['north_light'].values[:401]

n_steps = 400
n_inputs = 1
n_neurons = 200
n_layers = 4
n_outputs = 1


X_batch = data[-n_steps-1:-1].reshape(1, n_steps, n_inputs)
X_mean = X_batch.mean(axis=1).reshape(1)
X_std = X_batch.std(axis=1).reshape(1)
X_batch = (X_batch - X_batch.mean(axis=1)) / X_batch.std(axis=1)

y_batch = data[-n_steps:].reshape(1, n_steps, n_outputs)
y_batch = (y_batch - y_batch.mean(axis=1)) / y_batch.std(axis=1)

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='x')
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')


#DEEP RNN
with tf.name_scope('lstm'):
    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

learning_rate = 0.001

# Transform the output from [none, 1000, 200] to [none, 1000, 1] by feeding
# it through a fully connected final layer
with tf.name_scope('outputprojection'):
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons], name='rnn')
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs, name='stacked_outputs')
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

with tf.name_scope('eval'):
    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations = 20001

##Train the model
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    init.run()
    #saver.restore(sess, './model/my_time_series_model')
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 50 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            plt.figure(figsize=(7,3))
            plt.plot(X_batch[0][1:])
            plt.plot(y_pred[0])
            plt.show()
            
        if iteration % 200 == 0 and iteration != 0:
            real_signal = list(raw_data['north_light'].reshape(-1,))
            sequence = list(X_batch[0].reshape(-1,))
            for iteration in range(400):
                x_gen = np.array(sequence[-n_steps:]).reshape(1,-1,1)
                y_gen = outputs.eval(feed_dict={X: x_gen})
                sequence.append(y_gen[0, -1, 0])
                
            # Un Standardize the data to match up with whats in the csv
            sequence = np.array(sequence)
            sequence = sequence * X_std + X_mean
                
            plt.figure(figsize=(14,6))
            plt.plot(np.arange(len(sequence)), sequence, "r-")
            
            plt.plot(np.arange(len(real_signal)), real_signal, "b-")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.show()
            saver.save(sess, "./model/my_time_series_model", global_step=iteration)