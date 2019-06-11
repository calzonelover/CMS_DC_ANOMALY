import tensorflow as tf
import os
import abc

class BaseModel:
    def __init__(self, summary_dir="model/summary", save_dir="model_repo", log_dir="log_dir", model_name="model_name", save_checkpoint_time=0.1):
        # base
        self.save_checkpoint_time = save_checkpoint_time
        self.save_dir = save_dir
        self.model_name = model_name
        self.summary_dir = summary_dir
        self.log_dir = log_dir
        # session and log
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
    # fundamental func
    def init_variables(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def save(self):
        with self.graph.as_default():
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=self.save_checkpoint_time)
            saver.save(self.sess, os.path.join(self.save_dir, self.model_name))
    def restore(self):
        with self.graph.as_default():
            saver = tf.train.Save()
            saver.restore(self.sess, os.path.join(self.save_dir, self.model_name))
    @abc.abstractmethod
    def inference(self, x):
        return
    @abc.abstractmethod
    def predict(self, **kargs):
        return
    @abc.abstractmethod
    def train(self, x_batch, y_label_batch):
        pass
    # utility
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def fc_layer(self, input, W , size):
        b = self.weight_variable([size])
        return tf.add(tf.matmul(input, W), b)
    def PReLU(self, x, name):
        alpha = tf.get_variable(name, shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.add(tf.maximum(0.0, x),  tf.multiply(alpha, tf.minimum(0.0, x)))
    # sparse
    def l1_regularization(self, w):
        return tf.reduce_sum(tf.abs(w))
    def sparse_loss(self, w):
        return tf.reduce_sum([ self.l1_regularization(w_i) for w_i in w ])
    # contractive
    def contractive_loss(self, h, x):
        ## case PReLu activation fn
        pass
        ## only in case of sigmoid activation function
        # w = tf.transpose(w)
        # dh = tf.matmul(h, tf.subtract(1, h))
        # sum_i_w2 = tf.reduce_sum(tf.square(w), axis=1)
        # return tf.reduce_sum( tf.matmul(tf.square(dh), sum_i_w2), axis=1)