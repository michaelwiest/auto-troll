from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import numpy as np
import sys
import os
import csv



class EncoderDecoder(nn.Module):
    '''
    Model for predicting OTU counts of microbes given historical data.
    Uses fully connected layers and an LSTM (could use 1d convolutions in the
    future for better accuracy).

    As of now does not have a "dream" function for generating predictions from a
    seeded example.
    '''
    def __init__(self, hidden_dim, tweet_handler,
                 num_lstms,
                 use_gpu=False):
        super(EncoderDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.tweet_handler = tweet_handler

        self.num_lstms = num_lstms
        self.encoder = nn.LSTM(1,
                               hidden_dim, self.num_lstms)
        self.decoder_forward = nn.LSTM(1,
                                       hidden_dim, self.num_lstms)
        self.decoder_backward = nn.LSTM(1,
                                        hidden_dim, self.num_lstms)


        # Expansion layers from reduced number to raw number of strains
        self.after_lstm_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.tweet_handler.vocab_size),
            # nn.BatchNorm1d(self.tweet_handler.vocab_size)
            # nn.ReLU()
        )
        self.after_lstm_backward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.tweet_handler.vocab_size),
            # nn.BatchNorm1d(self.tweet_handler.vocab_size)
            # nn.Tanh()
        )
        self.lin_intermed = nn.Linear(self.hidden_dim, 1)

        # Non-torch inits.
        self.use_gpu = use_gpu
        self.hidden = None


    def forward(self, input_data, teacher_data=None):
        if teacher_data is not None:
            tf = teacher_data[0].transpose(0, 2)
            tb = teacher_data[1].transpose(0, 2)
        # Teacher data should be a tuple of length two where the first value
        # is the data corresponding to the future prediction and the
        # second value is the data corresponding to the reversed input.
        # data is shape: sequence_size x batch x num_strains
        input_data = input_data.transpose(0, 2)
        num_predictions = input_data.size(0)

        id = input_data #.contiguous().view(-1, self.batch_size, 1)
        # print(id.size())
        _, self.hidden = self.encoder(id, self.hidden)


        forward_hidden = self.hidden
        backward_hidden = self.hidden

        # Get the last input example.
        # print(input_data.size())
        forward_inp = input_data[-1, ...].unsqueeze(0) #.contiguous().view(-1, self.batch_size, 1)
        backward_inp = input_data[-1, ...].unsqueeze(0) #.contiguous().view(-1, self.batch_size, 1)
        # print(forward_inp.size())
        # print('$$')
        for i in range(num_predictions):
            # print(forward_inp.size())
            forward, forward_hidden = self.decoder_forward(forward_inp,
                                                           forward_hidden)
            backward, backward_hidden = self.decoder_backward(backward_inp,
                                                              backward_hidden)
            # print(forward.size())
            # forward = self.after_lstm_forward(forward)
            # backward = self.after_lstm_backward(backward)
            # forward = self.lin_intermed(forward)
            # backward = self.lin_intermed(backward)
            # print(forward.size())
            # print('---')
            # Add our prediction to the list of predictions.
            if i == 0:
                forward_pred = forward
                backward_pred = backward
            else:
                forward_pred = torch.cat((forward_pred,
                                          forward), 0)
                backward_pred = torch.cat((backward_pred,
                                          backward), 0)

            # If there is no teacher data then use the most recent prediction
            # to make the next prediction. Otherwise use the teacher data.
            if teacher_data is None:
                forward_inp = forward
                backward_inp = backward
            else:
                forward_inp = tf[i, ...].unsqueeze(0)
                backward_inp = tb[i, ...].unsqueeze(0)
        print(forward_pred.size())
        forward_pred = self.after_lstm_forward(forward_pred)
        return forward_pred.transpose(1, 2).transpose(0, 2), backward_pred.transpose(1, 2).transpose(0, 2)

    def __init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim).cuda()),
                           Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim)),
                           Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim))
                           )
    def add_cuda_to_variable(self, tensor, requires_grad=True):
        tensor = torch.FloatTensor(tensor)
        if self.use_gpu:
            return Variable(tensor.cuda(), requires_grad=requires_grad)
        else:
            return Variable(tensor, requires_grad=requires_grad)

    def get_intermediate_losses(self, loss_function, length,
                                teacher_force_frac,
                                num_batches=10):
        '''
        This generates some scores
        '''
        self.eval()

        train = [True, False]

        losses = []

        for i, is_train in enumerate(train):

            loss = 0
            for b in range(num_batches):
                # Select a random sample from the data handler.
                data, forward_targets = self.tweet_handler.get_N_samples_and_targets(self.batch_size,
                                                                                   length=length,
                                                                                   offset=length,
                                                                                   train=is_train)
                # this is the data that the backward decoder will reconstruct
                backward_targets = np.flip(data, axis=2).copy()
                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                data = self.add_cuda_to_variable(data).transpose(1, 2).transpose(0, 1)
                forward_targets = self.add_cuda_to_variable(forward_targets,
                                                            requires_grad=False)
                backward_targets = self.add_cuda_to_variable(backward_targets,
                                                            requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (forward_targets.transpose(1, 2).transpose(0, 1),
                          backward_targets.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(data,
                                                             teacher_data=tf)

                # We want to get the loss on a per-strain basis.

                floss = loss_function(forward_preds, forward_targets)
                bloss = loss_function(backward_preds, backward_targets)
                loss += floss + bloss

        if self.use_gpu:
            losses.append(loss.data.cpu().numpy().item() / (2 * num_batches))
        else:
            losses.append(loss.data.numpy().item() / (2 * num_batches))
        return losses

    def __print_and_log_losses(self, new_losses, save_params,
                               instantiate=False # Overwrite tensor if first time.
                               ):
        '''
        This function joins the newest loss values to the ongoing tensor.
        It also prints out the data in a readable fashion.
        '''
        if instantiate:
            self.loss_tensor = np.expand_dims(new_losses, axis=-1)
        else:
            new_losses = np.expand_dims(new_losses, axis=-1)
            self.loss_tensor = np.concatenate((self.loss_tensor, new_losses),
                                              axis=-1)

        to_print = self.loss_tensor[:, :, -1].sum(axis=1).tolist()
        print_str = ['Train loss: {}', '  Val loss: {}',
                     ' Test loss: {}']
        for i, tp in enumerate(to_print):
            print(print_str[i].format(tp))

        if save_params is not None:
            np.save(save_params[1], self.loss_tensor)


    def do_training(self,
                    slice_len,
                    batch_size,
                    epochs,
                    lr,
                    samples_per_epoch,
                    teacher_force_frac,
                    slice_incr_frequency=None, save_params=None):
        np.random.seed(1)

        self.batch_size = batch_size
        self.__init_hidden()

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # TODO: Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # For logging the data for plotting


        # Get some initial losses.
        # losses = self.get_intermediate_losses(loss_function, slice_len,
        #                                       teacher_force_frac)
        #
        # self.loss_tensor = None
        # self.__print_and_log_losses(losses, save_params, instantiate=True)

        for epoch in range(epochs):
            iterate = 0

            # For a specified number of examples per epoch. This basically
            # decides how many examples to do before increasing the length
            # of the slice of data fed to the LSTM.
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                self.train() # Put the network in training mode.

                # Select a random sample from the data handler.
                data, forward_targets = self.tweet_handler.get_N_samples_and_targets(self.batch_size,
                                                                                     length=slice_len,
                                                                                     offset=slice_len)
                # print(data.size())
                # print(forward_targets.size())
                # this is the data that the backward decoder will reconstruct
                backward_targets = np.flip(data, axis=2).copy()

                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                data = self.add_cuda_to_variable(data).transpose(1, 2).transpose(0, 1)
                forward_targets = self.add_cuda_to_variable(forward_targets,
                                                       requires_grad=False)
                backward_targets = self.add_cuda_to_variable(backward_targets,
                                                        requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (forward_targets.transpose(1, 2).transpose(0, 1),
                          backward_targets.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(data,
                                                             teacher_data=tf)

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.
                floss = loss_function(forward_preds, forward_targets)
                bloss = loss_function(backward_preds, backward_targets)
                loss = floss + bloss
                loss.backward()
                optimizer.step()
                iterate += 1

            print('Completed Epoch ' + str(epoch))

            # Get some train and val losses. These can be used for early
            # stopping later on.
            losses = self.get_intermediate_losses(loss_function, slice_len,
                                                  teacher_force_frac)
            self.__print_and_log_losses(losses, save_params)

            # If we want to increase the slice of the data that we are
            # training on then do so.
            if slice_incr_frequency is not None:
                if slice_incr_frequency > 0:
                    if epoch != 0 and epoch % slice_incr_frequency == 0:
                        slice_len += 1
                        # Make sure that the slice doesn't get longer than the
                        # amount of data we can feed to it. Could handle this with
                        # padding characters.
                        slice_len = min(self.tweet_handler.min_len - 1, int(slice_len))
                        print('Increased slice length to: {}'.format(slice_len))

            # Save the model and logging information.
            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                print('Saved model state to: {}'.format(save_params[0]))

    def daydream(self, primer, predict_len=100, window_size=20):
        '''
        Function for letting the Encoder Decoder "dream" up new data.
        Given a primer it will generate examples for as long as specified.
        '''
        if len(primer.shape) != 3:
            raise ValueError('Please provide a 3d array of shape: '
                             '(num_strains, slice_length, batch_size)')
        self.batch_size = primer.shape[-1]
        self.__init_hidden()
        self.eval()

        predicted = primer

        # Prime the model with all the data but the last point.
        inp = self.add_cuda_to_variable(predicted[:, :-1],
                                        requires_grad=False) \
            .transpose(0, 2) \
            .transpose(0, 1)[-window_size:, :, :]
        _, _ = self.forward(inp)
        for p in range(predict_len):

            inp = self.add_cuda_to_variable(predicted[:, -1, :]).unsqueeze(1)
            inp = inp.transpose(0, 2).transpose(0, 1)[-window_size:, :, :]
            # Only keep the last predicted value.
            output, _ = self.forward(inp)
            output = output[:, :, -1].transpose(0, 1).data
            if self.use_gpu:
                output = output.cpu().numpy()
            else:
                output = output.numpy()

            # Need to reshape the tensor so it can be concatenated.
            output = np.expand_dims(output, 1)
            # Add the new value to the values to be passed to the LSTM.
            predicted = np.concatenate((predicted, output), axis=1)

        return predicted
