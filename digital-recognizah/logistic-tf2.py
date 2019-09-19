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
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
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

    train_predictions = pd.get_dummies(mnist_data.iloc[:, 0], prefix="is")

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

    for i in range(args.steps):
        s = args.batch_size*(i+0)
        e = args.batch_size*(i+1)
        feeder = {
            x: mnist_data.iloc[s:e, 1:],
            y_actual: train_predictions.iloc[s:e],
        }
        train.run(feed_dict=feeder, session=sess)


    logging.warn("TF #2 Loss value = %f", sess.run(loss, feed_dict={
        x: mnist_data.iloc[:, 1:],
        y_actual: train_predictions,
    }))

    logging.warn("TF #2 Accuracy = %f", accuracy.eval(feed_dict={
        x: mnist_data.iloc[:, 1:],
        y_actual: train_predictions,
    }))
