#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import numpy as np
import tensorflow as tf
"""I wonder if tensorflow makes simple things easy for a version of vendor lock-in"""


if __name__ == "__main__":

    # Ex. 7


    # linear_model(x) = y
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)


    # Ex. 8 ... or shortcuts

    x2 = tf.placeholder(tf.float32, shape=[None, 4])
    y2 = tf.layers.dense(x2, units=1)


    # Ex. 9 ... data set with some named features + categorical features

    features = {
        "whatever": [[0.0], [1.0]],
        "sales": [[5.0], [10.0]],
        "department": ["whatever", "forever"],
    }

    department_column = tf.feature_column.categorical_column_with_vocabulary_list("department", ["whatever", "forever"])

    columns = [
        tf.feature_column.numeric_column("sales"),
        tf.feature_column.indicator_column(department_column)
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    # Before we run

    sess = tf.Session()


    # Feature 
    var_init = tf.global_variables_initializer()
    sess.run(var_init)
    tables_init = tf.tables_initializer()
    sess.run((var_init, tables_init))

    # Printing
    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
    print(sess.run(y2, {x2: [[0, 1, 2, 3], [9, 4, 5, 6]]}))

    # We out
    writer = tf.compat.v1.summary.FileWriter('events')
    writer.add_graph(tf.compat.v1.get_default_graph())
    writer.flush()
