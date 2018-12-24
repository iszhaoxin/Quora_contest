import numpy
import spacy
import random
import torch
from spacy.language import Language


class WordEmbedding:
    """
    Notes : 
        - Turn sentence-level dataset into sentence embeddings
        - The embedlist of on word can contain information of POS(external syntax label) and index(For attention-based method)
    Input
        - Sentence-level dataset :
            SentenceD = {corpus, dicVab, stat} -> dict
    Output
        - list of Word embeddings: (Save locally with pickle)
        
    Top-down data structure of output
        - Word embedding                    : sentEmbed = [(id, embedlist, lable)]  -> list
            - embedlist                     : [wordE & Pos & index]              -> list
    Top-down data structure of computational graph:
        - word embedding(if fine_tuned)
    """
    def __init__(self, params):
        self.params     = params
        
    def processing(self):
        if self.params.PCA:
            self._PCA()
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
    
    @property
    def embeddings(self):
        if self.params.trainable:
            return torch.from_numpy(...)

    def _word2vec(self):
        """
        Load word2vec from params.word
        """

    def _glove(self):
        """
        Load glove from params.word
        """

    def _fasttext(self):
            """
            Load fasttext from params.word
            """

    def _paragram(self):
            """
            Load paragram from params.word
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
    
