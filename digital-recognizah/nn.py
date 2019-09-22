#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import csv
import logging
import os
import random
import sys


import numpy as np
import pandas as pd
import tensorflow as tf


import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("solve mnist")
    parser.add_argument("--data", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.DEBUG)

    # Read csv locally
    df = pd.read_csv(args.data)
    X_features = df.iloc[:, 1:]/255.0
    Y_labels = pd.get_dummies(df.iloc[:, 0], prefix="digit")

    # Helper size variables
    num_examples, num_features = X_features.shape
    num_labels = Y_labels.shape[1]
    num_hidden_nodes = 300

    # Build out workflow
    x = tf.compat.v1.placeholder(tf.float32, [None, num_features])
    y = tf.compat.v1.placeholder(tf.float32, [None, num_labels])

    # Features -> Hidden
    W1 = tf.Variable(tf.random.normal([num_features, num_hidden_nodes], stddev=0.01), name="W1")
    b1 = tf.Variable(tf.random.normal([num_hidden_nodes]), name="b1")

    # Hidden -> Output
    W2 = tf.Variable(tf.random.normal([num_hidden_nodes, num_labels], stddev=0.01), name="W2")
    b2 = tf.Variable(tf.random.normal([num_labels]), name="b2")

    # Map from features to hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # Map from hidden layer to y
    y_predicted = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # Let's make predictions look neat
    y_clipped = tf.clip_by_value(y_predicted, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # Compute % matches
    num_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predicted, 1))
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32))

    # Initialize the environment
    init_vars = tf.compat.v1.global_variables_initializer()

    # Hyper parameters
    learning_rate = args.learning_rate
    iterations = args.iterations
    batch_size = args.batch_size

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cross_entropy)

    # Start the main show
    with tf.Session() as sess:
        sess.run(init_vars)
        total_batches = int(num_examples/batch_size)
        logging.warn("TOTAL BATCHES = %d (siqqq)", total_batches)

        for _ in range(args.iterations):
            avg_cost = 0.0

            for i in range(total_batches):

                start = batch_size*(i+0)
                end = batch_size*(i+1)

                batch_x = X_features.iloc[start:end, :]
                batch_y = Y_labels.iloc[start:end, :]

                _, cost = sess.run(
                    [train, cross_entropy],
                    feed_dict={x: batch_x, y: batch_y},
                )

                avg_cost += cost / total_batches

            logging.warning("Iterations %d: average cost = %f", i, avg_cost)

        accuracy_results = sess.run(
            accuracy,
            feed_dict={x: X_features, y: Y_labels},
        )

        logging.warning("Total accuracy = %f", accuracy_results)

        # Generate submission
        test_df = pd.read_csv(args.test)

        if args.demo:
            predictions = sess.run(
                tf.argmax(y_predicted, 1),
                feed_dict={x: test_df},
            )


            c = "yes"

            fig, axs = plt.subplots(5, 5)

            while c == "yes" or c == "y" or c == "":
                for i in range(5*5):
                    k = random.randint(0, len(predictions)-1)
                    x = i % 5
                    y = i // 5
                    f = axs[x, y].imshow(np.array(test_df.iloc[k, :]).reshape(28, 28))
                    axs[x, y].get_xaxis().set_visible(False)
                    axs[x, y].get_yaxis().set_visible(False)
                    # axs[x, y].set_title("Prediction = {}".format(predictions[k]))
                    for txt in axs[x, y].texts:
                        txt.set_visible(False)
                    axs[x, y].text(-5.5, 4.2, "{}".format(predictions[k]))

                fig.show()

                c = raw_input("Display more images (Y/n)? ").lower()
