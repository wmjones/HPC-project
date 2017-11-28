import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from tensorflow.python.ops import variable_scope

random.seed(0)
np.random.seed(0)
seq_max_len = 10
lr = 1e-3
total_episodes = 10
batch_size = 5
num_units = 10


# class Env:
#     def __init__(self):
#         self.total_reward = 0
#         self.sample = []

#     def generate_sample(self):
#         self.sample = [np.random.normal(0, 1, 2) for x in range(seq_max_len)]

#     def step(self, actions):
#         for i in range(len(actions)-1):
#             self.total_reward += LA.norm(self.sample[actions[i]] - self.sample[actions[i+1]])


# class Data:
#     def __init__(self, batch_size, seq_max_len):
#         self.batch_size = batch_size
#         self.seq_max_len = seq_max_len

#     def generate_batch(self):
#         s = []
#         A = []
#         G = []
#         for i in range(self.batch_size):
#             actions = [i for i in range(self.seq_max_len)]
#             random.shuffle(actions)
#             env = Env()
#             env.generate_sample()
#             env.step(actions)
#             s.append(env.sample)
#             A.append(actions)
#             G.append(env.total_reward)
#         return np.asarray(s, dtype=np.float32), np.asarray(A, dtype=np.float32), np.asarray(G, dtype=np.float32)


# init = tf.global_variables_initializer()
# data = Data(batch_size, seq_max_len)
# s_batch, A_batch, G_batch = data.generate_batch()
# s_batch = s_batch[:,:,0]
# s_batch = s_batch.reshape(-1, 10, 1)
# print()


# input_embed = tf.get_variable(
#         "input_embed", [1, 2, 2],
#         initializer=tf.random_uniform_initializer(1, 1.01))
# input_filter = tf.ones([1, 2, 2])
# tmp = tf.nn.conv1d(s_batch, input_filter, 1, "VALID")
# s = tf.placeholder(tf.float32, [None, seq_max_len, 1])
# A = tf.placeholder(tf.float32, [None, seq_max_len])
# G = tf.placeholder(tf.float32, [None, 1])
# seqlen = tf.placeholder(tf.int32, [None])

# tmp1 = tf.contrib.rnn.BasicRNNCell(10)
# tmp1 = tf.contrib.rnn.OutputProjectionWrapper(tmp1, output_size=1)
# tmp, _ = tf.nn.dynamic_rnn(tmp1, s, dtype=tf.float32)


class TimeSeriesData():
    def __init__(self, num_points, num_buckets):

        self.num_buckets = num_buckets
        self.num_points = num_points
        self.t_data = np.linspace(0, 1, num_points)
        self.x_data = [[self.t_data*np.cos(self.t_data[i]*2*np.pi), self.t_data[i]*np.sin(self.t_data[i]*2*np.pi)]
                       for i in range(len(self.t_data))]

    # def one_hot(self, data):
    #     x = data[0]
    #     y = data[1]
    #     out = np.zeros((self.num_buckets))
    #     if x > 0 and y > 0:
    #         out[0] = 1
    #     elif y > 0 and x <= 0:
    #         out[1] = 1
    #     elif y <= 0 and x < 0:
    #         out[2] = 1
    #     else:
    #         out[3] = 1
    #     return(out)

    def ret_true(self, t_series):
        return [[t_series[i]*np.cos(t_series[i]*2*np.pi), t_series[i]*np.sin(t_series[i]*2*np.pi)]
                for i in range(len(t_series))]

    def next_batch(self, batch_size, steps_in, steps_out):
        rand_start = np.random.rand(batch_size, 1)/2
        batch_ts = rand_start + np.arange(0.0, steps_in + steps_out)/self.num_points
        batch = np.asarray([self.ret_true(batch_ts[i]) for i in range(len(batch_ts))])
        batch_in = batch[:, :-steps_out, :]
        batch_out = batch[:, steps_in:, :]
        # batch_out = np.apply_along_axis(lambda x: self.one_hot(x), batch_out, 2)
        return(batch_in, batch_out)

num_buckets = 4
ts_data = TimeSeriesData(20, num_buckets)
batch_size = 10
hidden_dim = 30
output_dim = input_dim = 2
layers_stacked_count = 2
steps_in = 10
steps_out = 20

sample_x, sample_y = ts_data.next_batch(batch_size, steps_in, steps_out)
seq_length = sample_x.shape[1]
# # enc_inp = [tf.placeholder(tf.float32, [None, 2]) for t in range(seq_length)]
# # expected_sparse_output = [tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
# #                           for t in range(seq_length)]
enc_inp = tf.placeholder(tf.float32, [None, seq_length, 2])
expected_sparse_output = tf.placeholder(tf.float32, [None, seq_length, 2])
# output_embed = tf.contrib.layers.embed_sequence(y_true, vocab_size=num_buckets, embed_dim=4)

# # dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32)]
# # w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
# # b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
# # w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
# # b_out = tf.Variable(tf.random_normal([output_dim]))
# # reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp]

output_lengths = np.asarray([seq_length]*batch_size).astype(np.int32)
in_cells = []
for i in range(layers_stacked_count):
    with tf.variable_scope('RNN_{}'.format(i)):
        in_cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
in_cell = tf.nn.rnn_cell.MultiRNNCell(in_cells)
# in_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim), 2)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(in_cell, enc_inp, sequence_length=output_lengths, dtype=tf.float32)

out_cells = []
for i in range(layers_stacked_count):
    with tf.variable_scope('RNN_{}'.format(i)):
        out_cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, 2)

outputs = []
for i in range(steps_out):
    if i == 0:
        output, state = out_cell(tf.zeros((batch_size, 2)), out_cell.zero_state(batch_size=batch_size, dtype=tf.float32))
        # out_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    else:
        output, state = out_cell(expected_sparse_output[:, i, :], state)
    outputs.append(output)

outputs = tf.convert_to_tensor(outputs)
reshaped_outputs = []
for i in range(batch_size):
    reshaped_outputs.append(outputs[:, i, :])
outputs = tf.convert_to_tensor(reshaped_outputs)

predict_outputs = []
for i in range(steps_out):
    if i == 0:
        output, state = out_cell(tf.zeros((batch_size, 2)), out_cell.zero_state(batch_size=batch_size, dtype=tf.float32))
    else:
        output, state = out_cell(output, state)
    predict_outputs.append(output)

predict_outputs = tf.convert_to_tensor(predict_outputs)
predict_reshaped_outputs = []
for i in range(batch_size):
    predict_reshaped_outputs.append(predict_outputs[:, i, :])
predict_outputs = tf.convert_to_tensor(predict_reshaped_outputs)

# helper = tf.contrib.seq2seq.TrainingHelper(y_true, output_lengths)
# decoder = tf.contrib.seq2seq.BasicDecoder(
#     cell=out_cell,
#     helper=helper,
#     initial_state=out_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# )
# ouputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
# decoder_outputs, _, _, _ = tf.contrib.seq2seq.Decoder(encoder_outputs, initial_state=encoder_final_state)

# dec_inp = tf.zeros_like(enc_inp[0], dtype=np.float32)
# output, state = out_cell(dec_inp, encoder_final_state)
# outputs = [output]
# # for i in range(1, seq_length):
# #     if i > 0:
# #         variable_scope.get_variable_scope().reuse_variables()
# #     ouput, state = out_cell(outputs[i-1], state)
# #     outputs.append(output)
# # decoder_outputs, _ = tf.nn.dynamic_rnn(out_cell, expected_sparse_output, initial_state=encoder_final_state, dtype=tf.float32)
# # w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
# # reshaped_outputs = tf.map_fn(lambda x: (tf.matmul(x, w_out)), decoder_outputs)
# # reshaped_outputs = [(tf.matmul(i, w_out)) for i in decoder_outputs]  # + b_out
# # dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(reshaped_inputs, dec_inp, cell)   # dtype=tf.float32)
# # reshaped_outputs = [(tf.matmul(i, w_out) + b_out) for i in dec_outputs]

# # loss = 0
# # for _y, _y_hat in zip(decoder_outputs, expected_sparse_output):

# # decoder_outputs = np.asarray(outputs)
# # loss = tf.reduce_mean(tf.nn.l2_loss(np.asarray(outputs) - expected_sparse_output))

# # train = tf.train.AdamOptimizer(.001).minimize(loss)

learning_rate = 0.007  # Small lr helps not to diverge during training.
nb_iters = 150  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
# lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
lambda_l2_reg = 0  # L2 regularization of weights - avoids overfitting

reg_loss_1 = 0
for tf_var in tf.trainable_variables():
    reg_loss_1 += tf.reduce_mean(tf.nn.l2_loss(tf_var))
loss_1 = tf.reduce_mean(tf.nn.l2_loss(predict_outputs - expected_sparse_output)) + lambda_l2_reg*reg_loss_1
train_1 = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum).minimize(loss_1)

reg_loss_0 = 0
for tf_var in tf.trainable_variables():
    reg_loss_0 += tf.reduce_mean(tf.nn.l2_loss(tf_var))
loss_0 = tf.reduce_mean(tf.nn.l2_loss(outputs - expected_sparse_output)) + lambda_l2_reg*reg_loss_0
train_0 = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum).minimize(loss_0)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(3000):
        x_batch, y_batch = ts_data.next_batch(batch_size, seq_length)
        # x_batch = [x_batch[:, t, :] for t in range(seq_length)]
        # y_batch = [y_batch[:, t, :] for t in range(seq_length)]
        feed_dict = {enc_inp: x_batch, expected_sparse_output: y_batch}
        # print(sess.run(encoder_final_state, feed_dict=feed_dict))

    # tmp = sess.run(decoder_outputs, feed_dict=feed_dict)
    # print(tmp)
        # feed_dict = {enc_inp[t]: x_batch[t] for t in range(seq_length)}
        # feed_dict.update({expected_sparse_output[t]: y_batch[t] for t in range(seq_length)})
        # sess.run(train, feed_dict=feed_dict)
        # if i % 100 == 0:
        #     print(sess.run(loss, feed_dict=feed_dict))
        if i % 2 == 0:
            sess.run(train_0, feed_dict=feed_dict)
        else:
            sess.run(train_1, feed_dict=feed_dict)
        if i % 100 == 0:
            print(sess.run(loss_1, feed_dict=feed_dict))

    x, y = ts_data.next_batch(batch_size, steps_in, steps_out)
    feed_dict = {enc_inp: x, expected_sparse_output: y}
    outputs = sess.run(predict_outputs, feed_dict=feed_dict)

#     x_batch = [x[:, t, :] for t in range(seq_length)]
#     y_batch = [y[:, t, :] for t in range(seq_length)]
#     feed_dict = {enc_inp[t]: x_batch[t] for t in range(seq_length)}
#     # feed_dict.update({expected_sparse_output[t]: y_batch[t] for t in range(seq_length)})
#     # sess.run(reshaped_outputs, feed_dict=feed_dict)

#     feed_dict = {enc_inp[t]: x_batch[t] for t in range(seq_length)}
#     outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

for j in range(batch_size):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = x[j, :, k]
        expected = y[j, :, k]
        pred = outputs[j, :, k]

        label1 = "Seen (past) values" if k == 0 else "_nolegend_"
        label2 = "True future values" if k == 0 else "_nolegend_"
        label3 = "Predictions" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected)+len(past)),
                 expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()


# encoder_cell = tf.contrib.rnn.GridLSTMCell(num_units, [t])
# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
#     encoder_cell,
#     s,
#     # sequence_length=np.repeat([seq_max_len], batch_size),
#     dtype=tf.float32)            # Not sure about time major
# print(encoder_state)
# # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# # decoder_outputs, decoder_state = tf.nn.dynamic_rnn(
# #     decoder_cell,
# #     encoder_state,
# #     sequence_length=np.repeat([seq_max_len], batch_size),
# #     time_major=False,
# #     dtype=tf.float32)            # Not sure about time major


# with tf.Session() as sess:
#     # feed_dict = {s: s_batch, A: A_batch, G: G_batch}
#     sess.run(init)
#     feed_dict = {s: s_batch}
#     # sess.run(encoder_outputs, feed_dict=feed_dict)
#     out = sess.run(tmp, feed_dict=feed_dict)
#     print(out.shape)


# s_in = tf.unstack(s, seq_max_len, 1)
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(10)
# rnn_outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, s_in, dtype=tf.float32, sequence_length=seqlen)
# output = tf.layers.dense(inputs=rnn_outputs, units=10, activation=None)

# neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=A_t)
# loss = tf.reduce_mean(neg_log_prob * G_t)

# trainable_variables = tf.trainable_variables()
# gradient_holders = []

# for idx, var in enumerate(trainable_variables):
#     placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
#     gradient_holders.append(placeholder)

# gradients = tf.gradients(loss, trainable_variables)

# optimizer = tf.train.AdamOptimizer(lr)
# update_batch = optimizer.apply_gradients(zip(gradient_holders, trainable_variables))

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)

#     i = 0
#     gradBuffer = sess.run(tf.trainable_variables())
#     for ix, grad in enumerate(gradBuffer):
#         gradBuffer[ix] = grad*0

#     while i < total_episodes:
#         i += 1
#         # s_in = np.random.uniform(0, 1, N).reshape(-1, 1)
#         # G_t, A_t = reward_to_end(s_in, t)
#         # # print(A_t, G_t)
#         # s_in = np.array([s_in[t]], dtype=np.float32)
#         # r_in = np.array([G_t], dtype=np.float32)
#         # a_in = np.array([A_t], dtype=np.int32)
#         # feed_dict = {s: s_in, reward_holder: r_in, action_holder: a_in}
#         # grads = sess.run(gradients, feed_dict=feed_dict)
#         # for idx, grad in enumerate(grads):
#         #     gradBuffer[idx] += grad
#         # feed_dict = dict(zip(gradient_holders, gradBuffer))
#         # _ = sess.run(update_batch, feed_dict=feed_dict)


# test = model()
# test.generate_sample()
# test.step(range(3))
# print(test.sample)
# print(test.total_reward)


# SOLVE THE TSP PROBLEM
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
# def solve_tsp_dynamic(points):
#   #calc all lengths
#   all_distances = [[length(x,y) for y in points] for x in points]
#   #initial value - just distance from 0 to every other point + keep the track of edges
#   A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
#   cnt = len(points)
#   for m in range(2, cnt):
#     B = {}
#     for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
#       for j in S - {0}:
#         B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
#     A = B
#   res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
#   return np.asarray(res[1]) + 1 # 0 for padding
