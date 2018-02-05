import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

# Import and load the data into tensorflow
raw_data = pd.read_csv('north_south.csv')
X = raw_data[['north_queue_size', 'south_queue_size', 'east_queue_size', 'west_queue_size']]
y = raw_data['north_light']

X = X.as_matrix().astype(np.float32)
y = y.as_matrix().astype(np.float32).reshape(-1,1)

tf_X = tf.placeholder(dtype=tf.float32, shape=(None, 4))
tf_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))


# Create the Neural Network
n_hidden1 = 20
n_hidden2 = 10
n_hidden3 = 5
n_outputs = 1

with tf.name_scope('NeuralNetwork'):
    hidden1 = tf.layers.dense(tf_X, n_hidden1, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')

# Setup the gradient descent for training.
learning_rate = 0.003
with tf.name_scope('Train'):
    cost = tf.reduce_mean(tf.square(outputs - tf_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    training_op = optimizer.minimize(cost)
    
init = tf.global_variables_initializer()
n_epochs = 1

model_saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    model_saver.restore(sess,'./model.ckpt')
    for epoch in range(n_epochs):
        #sess.run(training_op, feed_dict={tf_X: X, tf_y: y})
        if epoch % 10000 == 0:
            epoch_cost = sess.run(cost, feed_dict={tf_X: X, tf_y: y})
            print(epoch, epoch_cost)
            model_saver.save(sess, './model.ckpt')
            
    y_pred = sess.run(outputs, feed_dict={tf_X: X})
    
# Plotting the accurate of the Deep net
plot_length = 200
y_pred_plot = y_pred[:plot_length].round().reshape(-1)
y_plot = y[:plot_length]
x_axis = np.arange(plot_length)
    
plt.figure(figsize=(14,4))
plt.plot(x_axis, y_pred_plot, 'r', label='DNN Prediction')
plt.plot(x_axis, y_plot, 'b', label='Simulator Output')
plt.ylabel('Light Status (1 is Green)')
plt.xlabel('Traffic Flow Operations')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, borderaxespad=0.)
plt.title('Neural Network Failover Trial 2')













