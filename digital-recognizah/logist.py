#!/usr/bin/env python3

"""
Roll your own logistic regression
...why not
"""


import argparse
import csv


def load_data(fi):
    """Return X, Y for mnist type data"""

    with args.train as fi:
        reader = csv.reader(fi)

        # Skip header
        next(reader, None)

        ys = []
        xs = []

        for blob in reader:
            ys.append(int(blob[0]))
            xs.append([int(x) for x in blob[1:]])

        assert len(ys) == len(xs)

    return xs, ys


def fit(xs, ys):
    """Return a "matrix" that "best fits" the data"""
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=argparse.FileType('r'))
    parser.add_argument("--test", required=True, type=argparse.FileType('w'))
    args = parser.parse_args()

    # ...
    xs, ys = load_data(args.train)

    print(xs)
