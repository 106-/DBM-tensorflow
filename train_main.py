#!/usr/bin/env python

import json
import argparse
import tensorflow as tf
import datetime
import os
from DBM import DBM
from sampling import oneshot_sampling
from mltools import LearningLog

parser = argparse.ArgumentParser("DBM learning script.", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-d", "--output_directory", action="store", type=str, default="./results/", help="directory to output parameter & log")
parser.add_argument("-s", "--filename_suffix", action="store", type=str, default=None, help="filename suffix")
args = parser.parse_args()

config = json.load(open(args.learning_config, "r"))
ll = LearningLog(config)

dtype = config["dtype"]

gen_dbm = DBM(config["generative-layers"], **config["generative-args"], dtype=dtype, gauss_init=True)
train_data = oneshot_sampling(gen_dbm, config["datasize"])[0]

optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002, epsilon=1e-8)

dbm = DBM(config["training-layers"], **config["training-args"], dtype=dtype)
dbm.train(args.learning_epoch, config["minibatch-size"], optimizer, train_data, gen_dbm, ll)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = [
    now,
    config["training-args"]["data_expectation"],
    config["training-args"]["model_expectation"]
]
if args.filename_suffix is not None:
    filename.append(args.filename_suffix)
filename.append("%s.json")
filename = "_".join(filename)

filepath = os.path.join(args.output_directory, filename)
ll.save(filepath%"log")

dbm.save(filepath%"model")