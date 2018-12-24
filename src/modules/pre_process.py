import os
import csv
import torch
import argparse
import collections
import numpy as np

class Dataset:
    """
    Input:
        - Path
    Output
        - corpus-level dataset : CLI
    Top-down data structure of output
        - Corpus-level dataset              : CorpusD   = {corpus, dicVab, stat}    -> dict
            - corpus                        : [i_1,...,i_n]                         -> list
                - items                     : item      = {ID, sentence, label}     -> dict
            - Statistical data of corpus    : stat      = {wordFre, TF-IDF, ...}    -> dict
            - Dictionary of Vocabulary      : dictVocab = {word2id, id2word}        -> dict    
    """
    def __init__(self):
        self.root   = os.path.dirname(os.path.realpath(__file__))+'/../data/'
        self.dataT   = collections.namedtuple('dataset', ['qid','question_text','target'])
        self.dataV   = collections.namedtuple('dataset', ['qid','question_text'])
    
    def process(self, fn):
        fn = self.root + fn
        dataset = []
        with open(fn, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for line in csv_reader:
                qid,question_text = line
                dataset.append(self.dataV(qid,question_text))
        return dataset

    def _dico(self):
        """
        Input   : corpus-level information 
        Output  : dicVocab = (word2id, id2word)
        """
        raise NotImplementedError
    
class PreProcess:
    """
    Notes
        - Change the structure of sentences and construct dictionaries of words
        - In case of using of Char-level information, word is preservated in output
    Input
        - Corpus-level dataset : CorpusD
    Output
        - Sentence-level dataset : SentenceD (Save locally with pickle)
    Top-down data structure of output
        - Sentence-level dataset            : SentenceD = {corpus, dicVab, stat}    -> dict
            - corpus                        : [i_1,...,i_n]                         -> list
                + items                     : item      = [(ID, sentence, label)]   -> list
                    + sentence              : [w_1,...,w_k] / [(w_i,d,w_j)]         -> list/dependency_graph
            - Statistical data of corpus    : stat      = {wordFre, TF-IDF, ...}    -> dict
            - Dictionary of Vocabulary      : dictVocab = {word2id, id2word}        -> dict
    """
    def __init__(self, params, dataset):
        self.params     = params
        self.dataset    = dataset

    def process(self):
        dataset         = self._split()
        if self.params.input_dropout:
            self._id            = self._input_dropout()
        if self.params.add_POS:
            self.POS            = self._add_POS()
        if self.params.add_position:
            self.add_position   = self._add_position()
        if self.params.add_dependency:
            self.add_dependency = self._add_dependency()

    def _split(self):
        """
        Input   : sentence level dataset 
        Output  : dicVocab = (word2id, id2word)
        """
        raise NotImplementedError

    def _add_noisy(self):
        raise NotImplementedError

    def _input_dropout(self):
        raise NotImplementedError

    def _add_POS(self):
        raise NotImplementedError

    def _add_position(self):
        raise NotImplementedError

    def _add_dependency(self):
        raise NotImplementedError


if __name__ == "__main__":   
    raise NotImplementedError