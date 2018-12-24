import numpy
import spacy
import random
import torch


class Predictor:
    """
    Notes : 
        - Classification model 
    Input : 
        - Word embedding                    : sentEmbed = [(id, embedlist, lable)]  -> list
        - 
    """
    def __init__(self, params):
        self.params     = params
        
    def process(self):
        if self.params.PCA:
            self._PCA()
        if self.params.fine_tuned:
            self._fine_tuned()
        if self.params.word2vec:
            _src_emb_word2vec   = self._word2vec()
        if self.params.glove:
            _src_emb_glove      = self._glove()
        if self.params.fasttext:
            _src_emb_fasttext   = self._fasttext()
        if self.params.paragram:
            _src_emb_paragram   = self._paragram()
        if self.params.concat:
            self._concat()
    
    def _word2vec(self):
        """
        Load word2vec from params.word
        """

    def _concat(self, *arg):
        """
        Notes : paramters
            - 'w' : word2vec 
            - 'g' : glove
            - 'f' : fasttext
            - 'pg' : paragram
            - 'pos' : POS
            - 'i' : postion
        """
        raise NotImplementedError
    
