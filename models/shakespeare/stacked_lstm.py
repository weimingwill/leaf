import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

is_v1 = True if tf.__version__.startswith('1') else False
if is_v1:
    from tensorflow.contrib import rnn
else:
    from tensorflow.compat.v1.nn import rnn_cell as rnn

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        init_range = 0.1
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])

        if is_v1:
            stacked_lstm = rnn.MultiRNNCell(
                [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        else:
            initializer = tf.random_uniform_initializer(-1.0, 1.0)
            stacked_lstm = rnn.MultiRNNCell(
                [rnn.LSTMCell(self.n_hidden, initializer=initializer) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:,-1,:],
                               units=self.num_classes,
                               kernel_initializer=tf.random_uniform_initializer(-init_range, init_range),
                               bias_initializer=tf.zeros_initializer())
        
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch
