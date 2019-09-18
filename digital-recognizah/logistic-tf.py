#!/usr/bin/env python


import argparse
import logging


import pandas as pd
import tensorflow as tf
import matplotlib as plt
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()


    # Logging setup
    logging.basicConfig(level=logging.DEBUG)



    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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


    # ITerate

    steps_number = 5000
    batch_size = 100

    for i in range(steps_number):
        input_batch, labels_batch = mnist.train.next_batch(batch_size)
        print("<<<")
        print(labels_batch)
        print(">>>")
        feeder = {
            x: input_batch,
            y_actual: labels_batch,
        }
        train.run(feed_dict=feeder)

    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images,
        y_actual: mnist.test.labels,
    })

    print(test_accuracy)
