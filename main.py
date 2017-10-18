import numpy as np
import tensorflow as tf
import scipy


def generate_data(m, d):
    # generate an array for a grid of m equidistant
    # points per dimension for the hypercube [0, 4pi]^d
    data = np.zeros((m**d, d))
    for i in range(0, d):
        data[:, i] = np.tile(np.repeat(np.linspace(0, 1, m), m**(d-(i+1))), m**i)
    return(data)


def generate_label(data):
    return(np.apply_along_axis(lambda x: sum(np.sin(x*4*np.pi)), 1, data))


# def model(m, d, num_of_layers, num_of_nodes):

m = 10
d = 2
num_of_nodes = 10
batch_size = 10000
train_data = generate_data(m, d)
train_label = generate_label(train_data)
eval_data = generate_data(2*m, d)
eval_label = generate_label(eval_data)
X = tf.placeholder(tf.float32, shape=[None, d])
y_true = tf.placeholder(tf.float32, shape=[None, 1])
actf = tf.nn.relu
hidden1 = tf.layers.dense(inputs=X, units=num_of_nodes, activation=actf)
hidden2 = tf.layers.dense(inputs=hidden1, units=num_of_nodes, activation=actf)
hidden3 = tf.layers.dense(inputs=hidden2, units=num_of_nodes, activation=actf)
hidden4 = tf.layers.dense(inputs=hidden3, units=num_of_nodes, activation=actf)
output = tf.layers.dense(inputs=hidden4, units=1)
loss = tf.losses.mean_squared_error(y_true, output)
optimizer = tf.train.AdamOptimizer(.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 10000

    for i in range(batches):
        rand_ind = np.random.randint(len(train_data), size=batch_size)
        sess.run(train, feed_dict={X: train_data[rand_ind],
                                   y_true: train_label[rand_ind].reshape(-1, 1)})
    pred = output.eval(feed_dict={X: eval_data})
    print(np.sqrt(sum((pred - eval_label.reshape(-1, 1))**2)))
    data = np.concatenate((eval_data, pred), axis=1)
    np.savetxt("model_data.csv", data)


    # NetworkStructure = [num_of_nodes]*num_of_layers
    # batch_size = 10

    # eval_data = generate_data(2*m, d)
    # eval_label = generate_label(eval_data)

    # feature_columns = [tf.feature_column.numeric_column("x", shape=[d])]
    # estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns,
    #                                       hidden_units=NetworkStructure)

    # input_fn = tf.estimator.inputs.numpy_input_fn(
    #     {"x": train_data}, train_label, batch_size=batch_size, num_epochs=None, shuffle=True)
    # # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    # #     {"x": train_data}, train_label, batch_size=batch_size, num_epochs=10, shuffle=False)
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     {"x": eval_data}, eval_label, batch_size=batch_size, num_epochs=10, shuffle=False)
    # # pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    # #     {"x": eval_data}, shuffle=False)

    # t_end = time.time() + 60
    # while time.time() < t_end:
    #     estimator.train(input_fn=input_fn, steps=30000)

    # # train_metrics = estimator.evaluate(input_fn=train_input_fn, steps=100)
    # eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=100)
    # preds = list(estimator.predict(input=pred_input_fn))
    # predictions = [p[''][0] for p in preds]
    # return(eval_metrics)


# def collect_data():
#     # data = np.zeros((1, 6))
#     # for k in range(0, 5):
#         # for j in range(3, 10):
#     for i in range(11, 20):
#         m = 10
#         d = 2
#         num_of_layers = 4
#         num_of_nodes = i
#         eval_metrics = model(m, d, num_of_layers, num_of_nodes)
#         f = open("network_data.csv", "ab")
#         np.savetxt(f, np.array([[m, d, num_of_layers, num_of_nodes,
#                                  eval_metrics['average_loss'], eval_metrics['loss']]]))
#         f.close()
#         # not sure which one i need to use
#         # tf.reset_default_graph()
#         tf.global_variables_initializer()


# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # save_path = saver.save(sess, "./models/model.ckpt")
#     collect_data()
    # np.savetxt("network_data.csv", data[1:len(data), :])
    # pdb.set_trace()

# train_metrics, eval_metrics = model(10, 2, 4, 8)
# print("train metrics: {}".format(train_metrics))
# print("eval metrics: {}".format(eval_metrics))

# pred = estimator.predict(input_fn=pred_input_fn)
# pred_for_csv = np.zeros((len(eval_data), 1))

# l2_error = 0
# for i, p in enumerate(pred):
#     # l2_error += (p['predictions'][0] - eval_label[i])**2
#     pred_for_csv[i] = p['predictions'][0]



# print(np.sqrt(l2_error))
# data = np.append(eval_data, pred_for_csv, 1)
# np.savetxt("model_data.csv", data)


# main()
