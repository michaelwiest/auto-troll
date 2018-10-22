from tweet_handler import TweetHandler
import os
import sys
import argparse


# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str,
                    help="The directory of input training data.")
parser.add_argument("-v", "--vocab", type=str,
                    help="The file with the vocab characters.")
args = parser.parse_args()
input_dir = args.data
vocab_file = args.vocab
files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.endswith('.csv')]

TH = TweetHandler(files, vocab_file)
TH.set_train_split()
TH.get_N_samples_and_targets(10, 5, 2)
TH.remove_urls()
