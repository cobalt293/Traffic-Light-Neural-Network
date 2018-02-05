import tensorflow as tf


class TrafficLightNeuralNet(object):
    def __init__(self, log_dir):
        #LogDirectory
        self.log_dir = log_dir


        # Model HyperParameters
        self.learning_rate = 0.0003
        self.n_steps = 100
        self.n_inputs = 1
        self.n_neurons = 200
        self.n_layers = 4
        self.n_outputs = 2

        with tf.name_scope('inputs'):
            self.X_tf = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
            self.y_tf = tf.placeholder(tf.int32, [None])

        with tf.name_scope('lstm'):
            self.layers = [tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons) for layer in range(self.n_layers)]
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.layers)
            self.rnn_outputs, self.states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X_tf, dtype=tf.float32)

        with tf.name_scope('evaluation'):
            self.logits = tf.layers.dense(self.states[-1][0], self.n_outputs)
            self.output_class = tf.argmax(self.logits, axis=1)
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_tf, logits=self.logits)
            self.loss = tf.reduce_mean(self.xentropy)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training_op = self.optimizer.minimize(self.loss)

            self.correct = tf.nn.in_top_k(self.logits, self.y_tf, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        
        
        self.merged_summaries = tf.summary.merge_all()
        self.writer_train = tf.summary.FileWriter(self.log_dir + '/model/train/')
        self.writer_test = tf.summary.FileWriter(self.log_dir + '/model/test/')
        self.saver = tf.train.Saver()

    def train(self, X_train, y_train, X_test, y_test, n_epochs=100):
        """Trains the LSTM given the input training data"""
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            init.run()
            for epoch in range(n_epochs):
                # Calculate Next Gradient Descent Step
                feed_dict = {self.X_tf: X_train, self.y_tf: y_train}
                summary, _ = sess.run([self.merged_summaries, self.training_op], feed_dict=feed_dict)
                self.writer_train.add_summary(summary, epoch)

                # Log Accuracy of Test Data
                feed_dict = {self.X_tf: X_test, self.y_tf: y_test}
                summary, acc = sess.run([self.merged_summaries, self.accuracy], feed_dict=feed_dict)
                self.writer_test.add_summary(summary, epoch)

                acc_train = self.accuracy.eval(feed_dict={self.X_tf: X_train, self.y_tf: y_train})
                acc_test = self.accuracy.eval(feed_dict={self.X_tf: X_test, self.y_tf: y_test})
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
               
                if acc_test > 0.93:
                    print("Accuracy has reached above 93%")
                    break
            #Save the final model
            self.saver.save(sess, self.log_dir + '/model/saved_model')

    def predict(self, X_pred):
        """predicts the next state of each bach sample"""
        # Predicting
        with tf.Session() as sess:
            self.saver.restore(sess, self.project_path + '/model/saved_model')
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

    #def delete_existing_model_files():
