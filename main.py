import numpy as np
import tensorflow as tf


def generate_data(m, d):
    # generate an array for a grid of m equidistant
    # points per dimension for the hypercube [0, 4pi]^d
    data = np.zeros((m**d, d))
    for i in range(0, d):
        data[:, i] = np.tile(np.repeat(np.linspace(0, 4*np.pi, m), m**(d-(i+1))), m**i)
    return(data)


def generate_label(data):
    return(np.apply_along_axis(lambda x: sum(np.sin(x)), 1, data))


def main():
    m, d = 20, 2
    train_data = generate_data(m, d)
    train_label = generate_label(train_data)
    eval_data = generate_data(2*m, d)
    eval_label = generate_label(eval_data)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[d])]
    estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                          hidden_units=[20, 20, 20, 20])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": train_data}, train_label, num_epochs=None, shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": eval_data}, eval_label, num_epochs=1, shuffle=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": eval_data}, num_epochs=1, shuffle=False)

    estimator.train(input_fn=train_input_fn, steps=10000)
    ev = estimator.evaluate(input_fn=eval_input_fn)
    print(ev["average_loss"])
    pred = estimator.predict(input_fn=predict_input_fn)

    pred_for_csv = np.zeros((len(eval_data), 1))
    l2_error = 0
    for i, p in enumerate(pred):
        l2_error += (p['predictions'][0] - eval_label[i])**2
        pred_for_csv[i] = p['predictions'][0]

    print(np.sqrt(l2_error))
    data = np.append(eval_data, pred_for_csv, 1)
    np.savetxt("model_data.csv", data)

main()
