import numpy as np
import tensorflow as tf
import time

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, d=2):
        self.device = device
        self.model_name = model_name
        self.d = d

        self.learning_rate = Config.LEARNING_RATE

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    # to make it run on a few threads
                    config=tf.ConfigProto(
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4
                    )
                    # config=tf.ConfigProto(
                    #     allow_soft_placement=True,
                    #     log_device_placement=False,
                    #     gpu_options=tf.GPUOptions(allow_growth=True))
                )
                self._create_tensor_board()
                self.sess.run(tf.global_variables_initializer())

    def _create_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.d], name='X')
        self.y_r = tf.placeholder(tf.float32, [None, 1], name='Yr')
        tf.summary.scalar("batch_size", tf.shape(self.x)[0])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        num_of_nodes = 10
        actf = tf.nn.relu
        self.hidden1 = tf.layers.dense(inputs=self.x, units=num_of_nodes, activation=actf, name="fc1")
        self.hidden2 = tf.layers.dense(inputs=self.hidden1, units=num_of_nodes, activation=actf, name="fc2")
        self.hidden3 = tf.layers.dense(inputs=self.hidden2, units=num_of_nodes, activation=actf, name="fc3")
        self.hidden4 = tf.layers.dense(inputs=self.hidden3, units=num_of_nodes, activation=actf, name="fc4")
        self.output = tf.layers.dense(inputs=self.hidden4, units=1, name="output")

        with tf.name_scope("cost"):
            self.cost = tf.losses.mean_squared_error(self.y_r, self.output)
        tf.summary.scalar("cost", self.cost)
        with tf.name_scope("train"):
            self.opt = tf.train.AdamOptimizer(Config.LEARNING_RATE)
            self.train_opt = self.opt.minimize(self.cost, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, x):
        prediction = self.sess.run(self.output, feed_dict={self.x: x})
        return prediction

    def train(self, x, y_r, trainer_id):
        feed_dict = {self.x: x, self.y_r: y_r}
        summary, _ = self.sess.run([self.merged, self.train_opt], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, self.get_global_step())

    def _create_tensor_board(self):
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name + '%f' % time.time())
        self.log_writer.add_graph(self.sess.graph)
