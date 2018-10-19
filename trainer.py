from tweet_handler import TweetHandler
import os
import sys
import argparse


# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str,
                    help="The directory of input training data.")
args = parser.parse_args()
input_dir = args.data

files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

TH = TweetHandler(files)
TH.set_train_split()
TH.get_N_samples_and_targets(10, 5, 0)
