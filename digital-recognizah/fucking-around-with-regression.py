#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import numpy as np
import tensorflow as tf
"""I wonder if tensorflow makes simple things easy for a version of vendor lock-in"""


if __name__ == "__main__":
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    # Randomly initialized predictions
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    # ...so here {{optimizer}} must be able to compute the gradient of {{loss}}
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(10000):
        _, loss_value =  sess.run((train, loss))
        # print(loss_value)

    print(sess.run(y_pred))
    print(loss_value)

    # We out
    writer = tf.compat.v1.summary.FileWriter('events')
    writer.add_graph(tf.compat.v1.get_default_graph())
    writer.flush()
