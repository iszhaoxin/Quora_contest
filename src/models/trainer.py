from __future__ import absolute_import

import re
import os
import scipy
import torch
import inspect
import scipy.linalg
import torch.optim as optim
from logging import getLogger
from logging import getLogger
from torch.nn import functional as F
from torch.autograd import Variable



def get_optimizer(s):
    """
    Notes : Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class Trainer(object):

    def __init__(self, emb, model, params):
        """
        Initialize trainer script.
        """
        self.model      = model
        self.params     = params
        self.optimizer  = get_optimizer(params.optim)
    
    def train(self):
        """
        Main train loop 
        """
        for epoch in range(self.params.n_epochs):
            
            # do something 

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    def criterion(self):
        """
        Evaluate result of current model
        - F1, accuracy, ....
        """
        raise NotImplementedError

    def fine_tuning(self, dico_train):
        """
        fine_tuning tricks
        """
        raise NotImplementedError

    
    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        raise NotImplementedError
    
    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        raise NotImplementedError

    def reload_best(self):
        """
        Reload the best mapping.
        """
        raise NotImplementedError

    def export(self):
        """
        Export embeddings.
        """
        raise NotImplementedError