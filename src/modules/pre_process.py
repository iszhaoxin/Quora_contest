import os
import csv
import torch
import argparse
import collections
import numpy as np
import pandas as pd

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
    def __init__(self, params):
        self.params = params
        self.root   = os.path.dirname(os.path.realpath(__file__))+'/..'
        self.dataT   = collections.namedtuple('dataset', ['qid','question_text','target'])
        self.dataV   = collections.namedtuple('dataset', ['qid','question_text','target'])
    
    def train_set(self):
        train_df = pd.read_csv(self.params.paths['train_path'])
        train_sents = [sent.split(" ") for sent in train_df["question_text"].values]
        train_labels = train_df["target"].values
        return train_sents, train_labels

    def valid_set(self):
        val_df = pd.read_csv(self.params.paths['valid_path'])
        val_sents = [sent.split(" ") for sent in val_df["question_text"].values]
        val_labels = val_df["target"].values
        return val_sents, val_labels
        
    def process(self, fn, dataset='train'):

        fn = os.path.join(self.root, fn)
        dataset = []
        with open(fn, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for line in csv_reader:
                qid, question_text, target = line
                dataset.append(self.dataV(qid,question_text,target))
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
        return dataset
        
    def _split(self):
        """
        Input   : sentence level dataset 
        Output  : dicVocab = (word2id, id2word)
        """
        return self.dataset
        # raise NotImplementedError

    def _add_noisy(self):
        raise NotImplementedError

    def _input_dropout(self):
        raise NotImplementedError


if __name__ == "__main__":   
    raise NotImplementedError