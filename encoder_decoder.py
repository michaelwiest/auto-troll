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
            nn.Linear(self.hidden_dim, 1),
            # nn.BatchNorm1d(1)
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
            nn.Linear(self.hidden_dim, 1),
            # nn.BatchNorm1d(self.tweet_handler.vocab_size)
            # nn.Tanh()
        )
        self.lin_final_forward = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.tweet_handler.vocab_size)
            )
        self.lin_final_backward = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.tweet_handler.vocab_size)
            )

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

        id = input_data.transpose(0, 2).transpose(0, 1)
        _, self.hidden = self.encoder(id, self.hidden)

        forward_hidden = self.hidden
        backward_hidden = self.hidden

        # Get the last input example.

        forward_inp = input_data[-1, ...].unsqueeze(0).transpose(0, 2).transpose(0, 1)
        backward_inp = input_data[-1, ...].unsqueeze(0).transpose(0, 2).transpose(0, 1)
        for i in range(num_predictions):
            # print(forward_inp.size())
            forward, forward_hidden = self.decoder_forward(forward_inp,
                                                           forward_hidden)
            backward, backward_hidden = self.decoder_backward(backward_inp,
                                                              backward_hidden)
            forward = self.after_lstm_forward(forward)
            backward = self.after_lstm_backward(backward)
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
                forward_inp = tf[i, ...].unsqueeze(0).transpose(0, 2)
                backward_inp = tb[i, ...].unsqueeze(0).transpose(0, 2)
        forward_pred = self.lin_final_forward(forward_pred)
        backward_pred = self.lin_final_backward(backward_pred)
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
                data = self.tweet_handler.get_N_samples_and_targets(self.batch_size,
                                                                   length=length,
                                                                   offset=length,
                                                                   train=True)
                inputs_oh, targets_oh, inputs_cat, targets_cat = data

                # this is the data that the backward decoder will reconstruct
                backward_targets_cat = np.flip(inputs_cat, axis=2).copy()
                backward_targets_oh = np.flip(inputs_oh, axis=2).copy()
                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                inputs_oh = self.add_cuda_to_variable(inputs_oh).transpose(1, 2).transpose(0, 1)
                targets_cat = self.add_cuda_to_variable(targets_cat,
                                                        requires_grad=False)
                targets_oh = self.add_cuda_to_variable(targets_oh,
                                                            requires_grad=False)
                backward_targets_cat = self.add_cuda_to_variable(backward_targets_cat,
                                                            requires_grad=False)
                backward_targets_oh = self.add_cuda_to_variable(backward_targets_oh,
                                                            requires_grad=False)
                inputs_cat = self.add_cuda_to_variable(inputs_cat,
                                                       requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (inputs_cat.transpose(1, 2).transpose(0, 1),
                          backward_targets_cat.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(inputs_cat,
                                                             teacher_data=tf)

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.
                forward_preds = forward_preds.transpose(1, 2)
                backward_preds = backward_preds.transpose(1, 2)
                floss = 0
                bloss = 0
                for b in range(length):
                    floss += loss_function(forward_preds[b, ...], targets_cat[b, ...].squeeze(1).long())
                    bloss += loss_function(backward_preds[b, ...], backward_targets_cat[b, ...].squeeze(1).long())
                loss += floss + bloss

            if self.use_gpu:
                losses.append(loss.data.cpu().numpy().item() / (2 * num_batches))
            else:
                losses.append(loss.data.numpy().item() / (2 * num_batches))
        return losses

    def __print_and_log_losses(self, new_losses, save_params):
        train_l = new_losses[0]
        val_l = new_losses[1]
        self.train_loss_vec.append(train_l)
        self.val_loss_vec.append(val_l)
        print('Train loss: {}'.format(train_l))
        print('  Val loss: {}'.format(val_l))

        if len(new_losses) == 3:
            test_l = new_losses[2]
            self.test_loss_vec.append(test_l)
            print(' Test loss: {}'.format(test_l))

        if save_params is not None:
            with open(save_params[1], 'w+') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.train_loss_vec)
                writer.writerow(self.val_loss_vec)
                if len(new_losses) == 3:
                    writer.writerow(self.test_loss_vec)




    def do_training(self,
                    length,
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

        loss_function = nn.CrossEntropyLoss()
        # TODO: Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # For logging the data for plotting
        self.train_loss_vec = []
        self.val_loss_vec = []

        # Get some initial losses.
        losses = self.get_intermediate_losses(loss_function, length,
                                              teacher_force_frac)

        self.__print_and_log_losses(losses, save_params)

        for epoch in range(epochs):
            iterate = 0

            # For a specified number of examples per epoch. This basically
            # decides how many examples to do before increasing the length
            # of the slice of data fed to the LSTM.
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                self.train() # Put the network in training mode.

                # Select a random sample from the data handler.
                data = self.tweet_handler.get_N_samples_and_targets(self.batch_size,
                                                                   length=length,
                                                                   offset=length,
                                                                   train=True)
                inputs_oh, targets_oh, inputs_cat, targets_cat = data

                # this is the data that the backward decoder will reconstruct
                backward_targets_cat = np.flip(inputs_cat, axis=2).copy()
                backward_targets_oh = np.flip(inputs_oh, axis=2).copy()
                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                inputs_oh = self.add_cuda_to_variable(inputs_oh).transpose(1, 2).transpose(0, 1)
                targets_cat = self.add_cuda_to_variable(targets_cat,
                                                        requires_grad=False)
                targets_oh = self.add_cuda_to_variable(targets_oh,
                                                            requires_grad=False)
                backward_targets_cat = self.add_cuda_to_variable(backward_targets_cat,
                                                            requires_grad=False)
                backward_targets_oh = self.add_cuda_to_variable(backward_targets_oh,
                                                            requires_grad=False)
                inputs_cat = self.add_cuda_to_variable(inputs_cat,
                                                       requires_grad=True)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (inputs_cat.transpose(1, 2).transpose(0, 1),
                          backward_targets_cat.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(inputs_cat,
                                                             teacher_data=tf)

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.
                forward_preds = forward_preds.transpose(1, 2)
                backward_preds = backward_preds.transpose(1, 2)
                floss = 0
                bloss = 0

                for b in range(self.batch_size):
                    floss += loss_function(forward_preds[b, ...], targets_cat[b, ...].squeeze(1).long())
                    bloss += loss_function(backward_preds[b, ...], backward_targets_cat[b, ...].squeeze(1).long())
                loss = floss + bloss
                loss.backward()
                optimizer.step()
                iterate += 1

            print('Completed Epoch ' + str(epoch))

            # Get some train and val losses. These can be used for early
            # stopping later on.
            losses = self.get_intermediate_losses(loss_function, length,
                                                  teacher_force_frac)
            self.__print_and_log_losses(losses, save_params)

            # If we want to increase the slice of the data that we are
            # training on then do so.
            if slice_incr_frequency is not None:
                if slice_incr_frequency > 0:
                    if epoch != 0 and epoch % slice_incr_frequency == 0:
                        length += 1
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
