import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

# Read in Data and look at the first 400 datapoints on the north light
raw_data = pd.read_csv('log.csv')
data = raw_data['north_light'].values[:401]


# Hyper Parms for the LSTM
learning_rate = 0.0003
n_steps = 100
n_inputs = 1
n_neurons = 200
n_layers = 4
n_outputs = 2

# Populate the training data and perform train test split
X = np.zeros((300, 100, 1))
y = np.zeros(300)

for i in range(300):
    X[i] = data[i:i+n_steps].reshape(n_steps,-1)
    y[i] = data[i+n_steps+1]

randomized = np.arange(300)
np.random.shuffle(randomized)
X = X[randomized]
y = y[randomized].astype(int)

# split into train and test
train_stop = np.floor(len(X) * 0.8).astype(int) # the index where training data stops and testing data starts

X_train = X[:train_stop]
y_train = y[:train_stop]

X_test = X[train_stop:]
y_test = y[train_stop:]

with tf.name_scope('inputs'):
    X_tf = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y_tf = tf.placeholder(tf.int32, [None])


#DEEP RNN
with tf.name_scope('lstm'):
    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X_tf, dtype=tf.float32)


#states_concat = tf.concat(axis=1, values=states)
logits = tf.layers.dense(states[-1][0], n_outputs)
output_class = tf.argmax(logits, axis=1)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_tf, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y_tf, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
#tf.summary.scalar('xentropy', xentropy)
merged_summaries = tf.summary.merge_all()
writer_train = tf.summary.FileWriter('./model/train/')
writer_test = tf.summary.FileWriter('./model/test/')



n_epochs = 1000
saver = tf.train.Saver()

## TRAINING
#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    init.run()
#    for epoch in range(n_epochs):
#        summary, _ = sess.run([merged_summaries, training_op], feed_dict={X_tf: X_train, y_tf: y_train})
#        writer_train.add_summary(summary, epoch)
#
#
#        summary, acc = sess.run([merged_summaries, accuracy], feed_dict={X_tf: X_test, y_tf: y_test})
#        writer_test.add_summary(summary, epoch)
#
#
#        acc_train = accuracy.eval(feed_dict={X_tf: X_train, y_tf: y_train})
#        acc_test = accuracy.eval(feed_dict={X_tf: X_test, y_tf: y_test})
#        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
#        
#        if acc_test > 0.93:
#            print("Accuracy has reached above 95%")
#            break
#    
#    #Save the final model
#    saver.save(sess, './model/saved_model')


 # Predicting
with tf.Session() as sess:
    saver.restore(sess, './model/saved_model')
    data = raw_data['north_light'].values[:600]
    target = np.copy(data[200:600])
    test = np.copy(data[200:600])
    test[200:] = -1

    for i in range(201):
        test_tf = test[i+100:i+200].reshape(1,-1,1)
        next_value = sess.run([output_class], feed_dict={X_tf: test_tf})
        test[i+199] = next_value[0][0]

    plt.clf()
    plt.cla()
    plt.close()
    plt.figure(figsize=(15,10))
    plt.subplot(311)
    plt.title("Target Signal RNN is trying to match")
    plt.plot(np.arange(len(target)), target, 'green')
    
    plt.subplot(312)
    plt.title("RNN predicts last 200 timesteps")
    plt.axvline(x=200)
    plt.plot(np.arange(len(test)), test, 'blue')
    
    plt.subplot(313)
    plt.title("RNN Error (Target-Prediction)")
    plt.plot(np.arange(len(test)), target-test, 'black')
    plt.fill_between(np.arange(len(test)), 0, target-test, facecolor='red')
    plt.savefig('LSTM_Classifier_Output')
    plt.show()
