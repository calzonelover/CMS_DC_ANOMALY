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
            # Body Layer
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
    def log_summary(self, x_input_test, EP): # for log per EP only
        [summary,] = self.sess.run([self.merged, ], feed_dict={self.x: x_input_test})
        self.train_writer.add_summary(summary, EP)
    def get_loss(self, x):
        return self.sess.run([self.total_loss, self.loss_mse, self.loss_reg], feed_dict={self.x: x})




class ContractiveAutoencoder(BaseModel):
    def __init__(self, LAMBDA=1e-4, input_dim=[2806], batch_size=1024, learning_rate=1e-4, beta1=0.7, beta2=0.9,**kwargs):
        super(ContractiveAutoencoder, self).__init__(**kwargs)
        # parameters
        self.LAMBDA = LAMBDA
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        with self.graph.as_default():
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[None, input_dim[0]], name="x_input")
            # Body Layer
            with tf.name_scope("model_body"):
                '''
                !! Careful !!
                If want to change any activation function
                also require to derive the contractive loss
                (Future work: This suppose should be fix by tf.gradient)
                '''
                self.W1 = self.weight_variable([input_dim[0], 128])
                self.wxb1 = self.fc_layer(self.x, self.W1, 128)
                self.encoder1 = self.PReLU(self.wxb1, name="encoder1")

                self.W2 = self.weight_variable([128, 64])
                self.wxb2 = self.fc_layer(self.encoder1, self.W2, 64)
                self.encoder2 = self.PReLU(self.wxb2, name="encoder2")

                self.W3 = self.weight_variable([64, 16])
                self.wxb3 = self.fc_layer(self.encoder2, self.W3, 16)
                self.encoder3 = self.PReLU(self.wxb3, name="encoder3")

                self.W4 = self.weight_variable([16, 64])
                self.wxb4 = self.fc_layer(self.encoder3, self.W4, 64)
                self.decoder1 = self.PReLU(self.wxb4, name="decoder1")

                self.W5 = self.weight_variable([64, 128])
                self.wxb5 = self.fc_layer(self.decoder1, self.W5, 128)
                self.decoder2 = self.PReLU(self.wxb5, name="decoder2")

                self.W6 = self.weight_variable([128, input_dim[0]])
                self.decoder3 = tf.math.sigmoid(self.fc_layer(self.decoder2, self.W6, input_dim[0]), name="decoder3")
                self.y_out = self.decoder3
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("mean_square"):
                    self.loss_mse = tf.reduce_mean(tf.squared_difference(self.y_out, self.x))
                    tf.summary.scalar("Loss_mean_square", self.loss_mse)
                with tf.name_scope("contractive"):
                    self.loss_con = self.LAMBDA * tf.add_n([
                            self.contractive_loss_prelu("model_body", "encoder1", self.W1, self.wxb1),
                            self.contractive_loss_prelu("model_body", "encoder2", self.W2, self.wxb2),
                            self.contractive_loss_prelu("model_body", "encoder3", self.W3, self.wxb3),
                            self.contractive_loss_prelu("model_body", "decoder1", self.W4, self.wxb4),
                            self.contractive_loss_prelu("model_body", "decoder2", self.W5, self.wxb5),
                            self.contractive_loss_sigmoid(self.W6, self.decoder3),
                        ])
                    tf.summary.scalar("Loss_contractive", self.loss_con)
                self.total_loss = tf.add(self.loss_mse, self.loss_con, name="total_loss")
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
    def log_summary(self, x_input_test, EP): # for log per EP only
        [summary,] = self.sess.run([self.merged, ], feed_dict={self.x: x_input_test})
        self.train_writer.add_summary(summary, EP)
    def get_loss(self, x):
        return self.sess.run([self.total_loss, self.loss_mse, self.loss_con], feed_dict={self.x: x})



class VariationalAutoencoder(BaseModel):
    def __init__(self, input_dim=[2806], batch_size=1024, learning_rate=1e-4, beta1=0.7, beta2=0.9,**kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        # parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        with self.graph.as_default():
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[None, input_dim[0]], name="x_input")
            # Body Layer
            self.W1 = self.weight_variable([input_dim[0], 128])
            self.encoder1 = self.PReLU(self.fc_layer(self.x, self.W1, 128), name="encoder1")
            self.W2 = self.weight_variable([128, 64])
            self.encoder2 = self.PReLU(self.fc_layer(self.encoder1, self.W2, 64), name="encoder2")
            self.W3 = self.weight_variable([64, 32])
            self.encoding = self.PReLU(self.fc_layer(self.encoder2, self.W3, 32), name="encoder3")

            self.means = tf.slice(self.encoding, [0, 0], [self.batch_size, 16])
            self.sigmas = tf.slice(self.encoding, [0, 16], [self.batch_size, 16])
            self.sampling = self.get_sampling(self.means, self.sigmas)            

            self.W4 = self.weight_variable([16, 64])
            self.decoder1 = self.PReLU(self.fc_layer(self.sampling, self.W4, 64), name="decoder1")
            self.W5 = self.weight_variable([64, 128])
            self.decoder2 = self.PReLU(self.fc_layer(self.decoder1, self.W5, 128), name="decoder2")
            self.W6 = self.weight_variable([128, input_dim[0]])
            self.decoder3 = tf.math.sigmoid(self.fc_layer(self.decoder2, self.W6, input_dim[0]), name="decoder3")
            self.y_out = self.decoder3
            print(self.get_sampling.get_shape(), self.x.get_shape(), self.y_out.get_shape())
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("mean_square"):
                    self.loss_mse = tf.reduce_mean(tf.squared_difference(self.y_out, self.x))
                    tf.summary.scalar("Loss_mean_square", self.loss_mse)
                with tf.name_scope("KL_divergence"):
                    self.loss_kl = self.kl_divergence(self.means, self.sigmas)
                    tf.summary.scalar("Loss_KL", self.loss_kl)
                self.total_loss = tf.math.add(self.loss_mse, self.loss_kl, name="total_loss")
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
    def log_summary(self, x_input_test, EP): # for log per EP only
        [summary,] = self.sess.run([self.merged, ], feed_dict={self.x: x_input_test})
        self.train_writer.add_summary(summary, EP)
    def get_loss(self, x):
        return self.sess.run([self.total_loss, self.loss_mse, self.loss_kl], feed_dict={self.x: x})