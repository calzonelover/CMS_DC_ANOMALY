import tensorflow as tf
import os
import abc

class BaseAutoencoder:
    def __init__(self, summary_dir="model/summary", save_dir="model_repo", log_dir="log_dir", model_name="model_name", save_checkpoint_time=0.1, gpu_memory_growth=True):
        # base
        self.save_checkpoint_time = save_checkpoint_time
        self.save_dir = save_dir
        self.model_name = model_name
        self.summary_dir = summary_dir
        self.log_dir = log_dir
        # session and log
        if not gpu_memory_growth:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
        else:
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph, config=self.config)
    def __del__(self):
        self.sess.close()
        print("object {} deleted".format(self.model_name))
    # fundamental func
    def init_variables(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def save(self):
        with self.graph.as_default():
            try:
                saver = tf.train.Saver(keep_checkpoint_every_n_hours=self.save_checkpoint_time)
                saver.save(self.sess, os.path.join(self.save_dir,  self.model_name, "{}.ckpt".format(self.model_name)))
            except ValueError:
                os.makedirs("./{}/{}/".format(self.save_dir,  self.model_name), exist_ok=True)
                saver.save(self.sess, os.path.join(self.save_dir,  self.model_name,  "{}.ckpt".format(self.model_name)))
    def restore(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(self.save_dir, self.model_name,  "{}.ckpt".format(self.model_name)))
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
    def get_var(self, param_name):
        with self.graph.as_default():
            return tf.get_variable(param_name)
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
    @staticmethod
    def step_fn(x):
        return tf.maximum(0.0, tf.sign(x))
    def contractive_loss_prelu(self, scope_name, alpha_name, w, wxb):
        w = tf.transpose(w)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            dh = tf.add_n([
                    tf.multiply(tf.get_variable(alpha_name, [1]), self.step_fn(wxb)),
                    self.step_fn(wxb)
                ])
        sum_i_w2 = tf.reshape(tf.reduce_sum(tf.square(w), axis=1), shape=[-1, 1])
        # sum_i_w2 = tf.reduce_sum(tf.square(w), axis=1)
        return tf.reduce_mean(tf.matmul(tf.square(dh), sum_i_w2))
    def contractive_loss_sigmoid(self, w, h):
        w = tf.transpose(w)
        dh = tf.multiply(h, tf.subtract(1.0, h))
        sum_i_w2 = tf.reshape(tf.reduce_sum(tf.square(w), axis=1), shape=[-1, 1]) # [N, 1]
        return tf.reduce_mean( tf.matmul(tf.square(dh), sum_i_w2)) # [BS, N] x [N, 1] = [BS, 1]
    # Variational
    def get_sampling(self, means, sigmas):
        return tf.add(means, tf.multiply(sigmas, tf.random.normal(shape=tf.shape(sigmas))))
    def kl_divergence(self, means, sigmas):
        return tf.reduce_mean( tf.multiply(
                        0.5,
                        tf.reduce_sum(
                            tf.math.add_n(
                                [
                                    tf.math.square(means),
                                    tf.math.square(sigmas),
                                    tf.add(
                                        tf.multiply(
                                            -1.0,
                                            tf.math.log(tf.math.square(sigmas))
                                        ),
                                        -1.0
                                    )
                                ])
                        , axis=1)
                    ))



class BaseMalfunctionSpotter:
    def __init__(self,  model_name, **kwargs):
        self.save_checkpoint_time = kwargs.get('save_checkpoint_time', 0.1)
        self.save_dir = kwargs.get('save_dir', 'model_repo')
        self.model_name = kwargs.get('model_name')
        self.summary_dir = kwargs.get('summary_dir', 'model/summary')
        self.log_dir = kwargs.get('log_dir', 'log_dir')
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
            try:
                saver = tf.train.Saver(keep_checkpoint_every_n_hours=self.save_checkpoint_time)
                saver.save(self.sess, os.path.join(self.save_dir,  self.model_name, "{}.ckpt".format(self.model_name)))
            except ValueError:
                os.makedirs("./{}/{}/".format(self.save_dir,  self.model_name), exist_ok=True)
                saver.save(self.sess, os.path.join(self.save_dir,  self.model_name,  "{}.ckpt".format(self.model_name)))
    def restore(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(self.save_dir, self.model_name,  "{}.ckpt".format(self.model_name)))
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
    def get_var(self, param_name):
        with self.graph.as_default():
            return tf.get_variable(param_name)
    def fc_layer(self, input, W , size):
        b = self.weight_variable([size])
        return tf.add(tf.matmul(input, W), b)
    def PReLU(self, x, name):
        alpha = tf.get_variable(name, shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.add(tf.maximum(0.0, x),  tf.multiply(alpha, tf.minimum(0.0, x)))
    # miscellaneous
    @staticmethod
    def step_fn(x):
        return tf.maximum(0.0, tf.sign(x))