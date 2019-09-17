#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse


import numpy as np
import tensorflow as tf
"""I wonder if tensorflow makes simple things easy for a version of vendor lock-in"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser("solve mnist")
    parser.add_argument("--train-csv", required=True)
    args = parser.parse_args()
