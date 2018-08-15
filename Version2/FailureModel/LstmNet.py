import tensorflow as tf

# If something is wrong with GPU and you want to force use the CPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class FailureModel(object):
    def __init__(self, log_dir):
        #LogDirectory
        self.log_dir = log_dir

        # Model HyperParameters
        self.learning_rate = 0.001
        self.n_steps = 50
        self.n_inputs = 4
        self.n_neurons = 100
        self.n_layers = 3
        self.n_outputs = 2

        with tf.name_scope('inputs'):
            self.X_tf = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
            self.y_tf = tf.placeholder(tf.int32, [None])

        with tf.name_scope('lstm'):
            self.keep_prob = tf.placeholder_with_default(1.0, shape=())
            self.cells = [tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons) for layer in range(self.n_layers)]
            self.cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in self.cells]
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.cells_drop)
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
        self.writer_train = tf.summary.FileWriter(self.log_dir + '/training_performance/train/')
        self.writer_test = tf.summary.FileWriter(self.log_dir + '/training_performance/test/')
        self.saver = tf.train.Saver()

    def train(self, X_train, y_train, X_test, y_test, n_epochs=10):
        """Trains the LSTM given the input training data"""
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            init.run()
            for epoch in range(n_epochs):
                # Calculate Next Gradient Descent Step
                feed_dict = {self.X_tf: X_train, self.y_tf: y_train, self.keep_prob: 0.5}
                summary, _ = sess.run([self.merged_summaries, self.training_op], feed_dict=feed_dict)
                self.writer_train.add_summary(summary, epoch)

                # Log Accuracy of Test Data
                feed_dict = {self.X_tf: X_test, self.y_tf: y_test, self.keep_prob: 0.5}
                summary, acc = sess.run([self.merged_summaries, self.accuracy], feed_dict=feed_dict)
                self.writer_test.add_summary(summary, epoch)

                if epoch % 10 == 0:
                    acc_train = self.accuracy.eval(feed_dict={self.X_tf: X_train, self.y_tf: y_train, self.keep_prob: 1.0})
                    acc_test = self.accuracy.eval(feed_dict={self.X_tf: X_test, self.y_tf: y_test, self.keep_prob: 1.0})
                    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            #Save the final model
            self.saver.save(sess, self.log_dir + '/model')

    def predict(self, X_pred):
        """predicts the next state of each bach sample"""
        
        with tf.Session() as sess:
            self.saver.restore(sess, self.log_dir + '/model')

            y_pred = sess.run(self.output_class, feed_dict={self.X_tf: X_pred, self.keep_prob: 1.0})
            return y_pred[0]