#!/usr/bin/env python


import argparse
import logging


import pandas as pd
import tensorflow as tf
import matplotlib as plt
import numpy as np


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.DEBUG)

    # Read csv locally
    mnist_data = pd.read_csv(args.data)

    # ...
    num_examples = mnist_data.shape[0]
    num_features = mnist_data.shape[1]-1
    num_labels = 10

    # Get data into constants
    x = tf.placeholder(tf.float32, [None, num_features])
    y_actual = tf.placeholder(tf.float32, [None, num_labels])

    # ...
    W = tf.Variable(tf.zeros([num_features, num_labels]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([num_labels]), dtype=tf.float32)


    mod = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=y_actual))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)


    correct = tf.equal(tf.argmax(mod, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    # Start cooking

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.as_default()


    # ITerate

    steps_number = 5000
    batch_size = 100


    yyyy = sess.run(tf.one_hot(mnist_data.iloc[:, 0], 10, dtype=tf.float32))

    for i in range(1000):
        s = 100*(i+0)
        e = 100*(i+1)
        input_batch = mnist_data.iloc[s:e, 1:]
        labels_batch = sess.run(tf.one_hot(mnist_data.iloc[s:e, 0], 10, dtype=tf.float32))

        feeder = {
            x: input_batch,
            y_actual: labels_batch,
        }
        train.run(feed_dict=feeder, session=sess)
    print(sess.run(loss, feed_dict={
        x: mnist_data.iloc[:, 1:],
        y_actual: yyyy,
    }))

    test_accuracy = accuracy.eval(feed_dict={
        x: mnist_data.iloc[:, 1:],
        y_actual: yyyy,
    })

    print("TF #2 Accuracy =", test_accuracy)
