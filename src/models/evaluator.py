import numpy as np
from copy import deepcopy
from logging import getLogger
from torch.autograd import Variable
from torch import Tensor as torch_tensor

logger = getLogger()

class Evaluator(object):

    def __init__(self, params, val_sents, val_labels, model):
        """
        Initialize evaluator.
        """
        self.params = params
        self.model = model
        self.val_sents, self.val_labels = val_sents, val_labels
    
    def report(self):
        self.model.predictor.score(self.val_sents, self.val_labels)
    
    def f1(self):
        raise NotImplementedError
    
    def accuracy(self):
        raise NotImplementedError

    def prediction(self):
        raise NotImplementedError

    def recall(self):
        raise NotImplementedError

    def ROC(self):
        raise NotImplementedError

    def otherfuncs(self):
        raise NotImplementedError