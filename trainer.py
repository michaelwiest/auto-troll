from tweet_handler import TweetHandler
import os
import sys
from params import *
import argparse
from encoder_decoder import EncoderDecoder

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
TH.remove_urls()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

save_params = (os.path.join(output_dir, model_name),
               os.path.join(output_dir, log_name))

enc = EncoderDecoder(hidden_dim, TH, num_lstms)
enc.do_training(seq_len,
                batch_size,
                num_epochs,
                learning_rate,
                samples_per_epoch,
                teacher_force_frac,
                slice_incr_frequency=slice_incr_frequency,
                save_params=save_params)
