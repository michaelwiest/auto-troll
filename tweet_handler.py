import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split



'''
May want to implement this:
https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
to remove urls.
'''

def rand_line_start(string, length, pad_char='@'):
    start = np.random.randint(len(string))
    temp = string[start: start + length]
    temp += pad_char * (length - len(temp))
    return temp


class TweetHandler(object):
    def __init__(self, file_list, vocab_file, pad_char='\xe6',
                 sos_char='\xf7', eos_char='\xf5'):
        self.pad_char = pad_char
        self.file_list = file_list
        self.vocab_file = vocab_file
        self.__join_data_inputs()
        self.__build_vocabulary()

    def __build_vocabulary(self):
        self.vocab = open(self.vocab_file, 'r').read().splitlines()
        self.vocab_string = ''.join(self.vocab)
        self.n_letters = len(self.vocab)
        print('There are {} letters in the vocabulary'.format(self.n_letters))

    def __join_data_inputs(self):
        for i, f in enumerate(self.file_list):
            df = pd.read_csv(f)
            if i == 0:
                self.data = df
            else:
                self.data.append(df)

    def set_train_split(self, ratio=0.8):
        self.train, self.validation = train_test_split(self.data,
                                                       test_size=1 - ratio)

    def get_N_samples_and_targets(self, N, length, offset, train=True):
        if train:
            data = self.train
        else:
            data = self.validation
        sub = data.sample(N)

        # Get just the padded strings from the df.
        just_str = sub.content.apply(rand_line_start,
                                      args=(length, self.pad_char, )).values.tolist()

        # One hot encode the strings.
        out = torch.stack([self.line_to_tensor(s) for s in just_str]).squeeze(2)
        print(out.size())


    def letter_to_index(self, letter):
        letter = letter.lower()
        return self.vocab_string.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letter_to_index(letter)] = 1
        return tensor
