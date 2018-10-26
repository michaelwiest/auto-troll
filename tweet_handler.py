import pandas as pd
import numpy as np
import torch
import re
from sklearn.model_selection import train_test_split



def rand_line_start(string,
                    length,
                    pad_char,
                    target_offset):
    start = np.random.randint(len(string))
    input_string = string[start: start + length]
    input_string += pad_char * (length - len(input_string))
    target_string = string[start + target_offset:
                           start + target_offset + length]
    target_string += pad_char * (length - len(target_string))
    return pd.Series([input_string, target_string])

def prefix_suffix_line_with_chars(string, prefix_char, suffix_char):
    string = prefix_char + string + suffix_char
    return string

def strip_urls(string):
    # Crazy regex I found online for matching URLs
    regex_str = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»\"\"\'\']))'
    return re.sub(regex_str, '', string, flags=re.MULTILINE)

class TweetHandler(object):
    def __init__(self, file_list, vocab_file, pad_char='\xe6',
                 sos_char='\xf7', eos_char='\xf5'):
        self.pad_char = pad_char
        self.sos_char = sos_char
        self.eos_char = eos_char
        self.file_list = file_list
        self.vocab_file = vocab_file
        self.__join_data_inputs()
        self.__add_special_chars_to_data()
        self.__build_vocabulary()
        self.vec_index_lookup = np.vectorize(self.letter_to_index)


    def __build_vocabulary(self):
        '''
        Open the vocabulary file and set the pertinent fields.
        '''
        self.vocab = open(self.vocab_file, 'r').read().splitlines()
        self.vocab += [self.pad_char, self.sos_char, self.eos_char, '\x85']
        self.vocab_string = ''.join(self.vocab)
        self.vocab_size = len(self.vocab)
        print('There are {} letters in the vocabulary'.format(self.vocab_size))

    def __join_data_inputs(self):
        for i, f in enumerate(self.file_list):
            df = pd.read_csv(f)
            if i == 0:
                self.data = df
            else:
                self.data.append(df)

    def __add_special_chars_to_data(self):
        '''
        This adds the sos and eos characters to each string.
        '''
        if self.data is None:
            raise ValueError('Please build dataset first')
        self.data.content = self.data.content.apply(prefix_suffix_line_with_chars,
                                                    args=(self.sos_char,
                                                          self.eos_char,))
        self.data.content = self.data.content.str.lower()

    def remove_urls(self):
        if self.data is None:
            raise ValueError('Please build dataset first')
        self.data.content = self.data.content.apply(strip_urls)

    def set_train_split(self, ratio=0.8):
        self.train, self.validation = train_test_split(self.data,
                                                       test_size=1 - ratio)

    def get_N_samples_and_targets(self, N, length, offset, train=True):
        if self.train is None:
            raise ValueError('Please set train val split first.')
        if train:
            data = self.train
        else:
            data = self.validation
        sub = data.sample(N)

        # Get just the padded strings from the df.
        out = sub.content.apply(rand_line_start,
                                      args=(length,
                                            self.pad_char,
                                            offset)).values

        inputs = np.array([list(s) for s in out[:, 0]])
        targets = np.array([list(s) for s in out[:, 1]])

        inputs_category = torch.Tensor(self.vec_index_lookup(inputs)).unsqueeze(2)
        targets_category = torch.Tensor(self.vec_index_lookup(targets)).unsqueeze(2)

        # # One hot encode the strings.
        inputs_oh = torch.stack([self.line_to_tensor(s) for s in inputs]).squeeze(2)
        targets_oh = torch.stack([self.line_to_tensor(s) for s in targets]).squeeze(2)

        return inputs_oh, targets_oh, inputs_category, targets_category, targets


    def letter_to_index(self, letter):
        letter = letter.lower()
        return self.vocab_string.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.vocab_size)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.vocab_size)
        for li, letter in enumerate(line):
            tensor[li][0][self.letter_to_index(letter)] = 1
        return tensor
