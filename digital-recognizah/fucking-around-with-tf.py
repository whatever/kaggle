#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import numpy as np
import tensorflow as tf
"""I wonder if tensorflow makes simple things easy for a version of vendor lock-in"""


if __name__ == "__main__":

    # parser = argparse.ArgumentParser("solve mnist")
    # parser.add_argument("--train-csv", required=True)
    # args = parser.parse_args()

    # Ex 1.
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)
    c = tf.constant([[[1.0]], [[1.9]]])
    d = tf.constant([[1.0], [2.0]])

    # Ex. 3
    sess = tf.Session()
    res = sess.run({
        "nvm": a+d,
    })

    # Ex. 4
    random_vec = tf.random_uniform(shape=(3, 1, 10))
    scaled = 3*random_vec

    res = sess.run((random_vec, scaled))
    print(res)


    # Ex. 5 Feeding

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

    res = sess.run(z, feed_dict={x: [[[1.0]]], y: [[[2.0]]]})
    print(res)


    # Fin?
    writer = tf.compat.v1.summary.FileWriter('.')
    writer.add_graph(tf.compat.v1.get_default_graph())
    writer.flush()
