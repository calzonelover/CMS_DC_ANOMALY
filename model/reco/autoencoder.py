import tensorflow as tf
import numpy as np
import os

from model.NN.base import BaseModel

class SparseAutoencoder(BaseModel):
    def __init__(self, LAMBDA=1e-4, input_dim=[2806], batch_size=1024, learning_rate=1e-4, beta1=0.7, beta2=0.9,**kwargs):
        super(SparseAutoencoder, self).__init__(**kwargs)
        # parameters
        self.LAMBDA = LAMBDA
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        with self.graph.as_default():
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[None, input_dim[0]], name="x_input")
            # layer
            self.W1 = self.weight_variable([input_dim[0], 128])
            self.encoder1 = self.PReLU(self.fc_layer(self.x, self.W1, 128), name="encoder1")
            self.W2 = self.weight_variable([128, 64])
            self.encoder2 = self.PReLU(self.fc_layer(self.encoder1, self.W2, 64), name="encoder2")
            self.W3 = self.weight_variable([64, 16])
            self.encoder3 = self.PReLU(self.fc_layer(self.encoder2, self.W3, 16), name="encoder3")
            self.W4 = self.weight_variable([16, 64])
            self.decoder1 = self.PReLU(self.fc_layer(self.encoder3, self.W4, 64), name="decoder1")
            self.W5 = self.weight_variable([64, 128])
            self.decoder2 = self.PReLU(self.fc_layer(self.decoder1, self.W5, 128), name="decoder2")
            self.W6 = self.weight_variable([128, input_dim[0]])
            self.decoder3 = tf.math.sigmoid(self.fc_layer(self.decoder2, self.W6, input_dim[0]), name="decoder3")
            self.y_out = self.decoder3
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("mean_square"):
                    self.loss_mse = tf.reduce_mean(tf.squared_difference(self.y_out, self.x))
                    tf.summary.scalar("Loss_mean_square", self.loss_mse)
                with tf.name_scope("l1_regularization"):
                    self.loss_reg = self.LAMBDA * self.sparse_loss([self.W1, self.W2, self.W3, self.W4, self.W5, self.W6])
                    tf.summary.scalar("Loss_reg", self.loss_reg)
                self.total_loss = tf.add(self.loss_mse, self.loss_reg, name="total_loss")
                tf.summary.scalar("Total_loss", self.total_loss)
            # optimizer
            with tf.name_scope("Optimizer"):
                self.optimizer = tf.train.AdamOptimizer(
                    self.learning_rate, beta1=beta1, beta2=beta2,
                    ).minimize(self.total_loss)
            # summary
            self.train_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, self.model_name, self.log_dir), self.sess.graph)
            self.merged = tf.summary.merge_all()
    def train(self, x_batch):
        _, l = self.sess.run([self.optimizer, self.total_loss], feed_dict={self.x: x_batch})
        # print('loss ', l)
    def log_summary(self, x_input_test, EP): # for log per EP only
        [summary,] = self.sess.run([self.merged, ], feed_dict={self.x: x_input_test})
        self.train_writer.add_summary(summary, EP)
    def get_loss(self, x):
        return self.sess.run(self.total_loss, feed_dict={self.x: x})