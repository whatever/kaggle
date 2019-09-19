#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import numpy as np
import pandas as pd
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("solve mnist")
    parser.add_argument("--data", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    # Read csv locally
    df = pd.read_csv(args.data)
    X_features = df.iloc[:, 1:]
    Y_labels = pd.get_dummies(df.iloc[:, 0], prefix="digit")

    # Helper size variables
    num_examples, num_features = X_features.shape
    num_labels = Y_labels.shape[1]
    num_hidden_nodes = 300

    # Build out workflow
    x_actual = tf.compat.v1.placeholder(tf.float32, [None, num_features])
    y_actual = tf.compat.v1.placeholder(tf.float32, [None, num_labels])

    # Features -> Hidden
    W1 = tf.Variable(tf.random.normal([num_features, num_hidden_nodes]), name="W1")
    b1 = tf.Variable(tf.random.normal([num_hidden_nodes]), name="b1")

    # Hidden -> Output
    W2 = tf.Variable(tf.random.normal([num_hidden_nodes, num_labels]), name="W2")
    b2 = tf.Variable(tf.random.normal([num_labels]), name="b2")

    # Print setup!
    assert False, "Start here: you need to start training things!"
