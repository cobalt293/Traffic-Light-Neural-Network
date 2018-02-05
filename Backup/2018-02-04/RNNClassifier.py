## RNN Example that will classify MNIST Data


import tensorflow as tf

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

# Tensorflow input placeholders
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# Define RNN 
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
