import numpy
import spacy
import random
from spacy.language import Language


class Encoder:
    """
    Notes : 
        - Turn sentence-level dataset into sentence embeddings
        - The embedlist of on word can contain information of POS(external syntax label) and index(For attention-based method)
    Input
        - Sentence-level dataset :
            SentenceD = {corpus, dicVab, stat} -> dict
        - list of Word embeddings: 
    Output
        - sentence embedding or hidden state of predictor model (E.g. The output state of LSTM)
    
    Top-down data structure of output
        - sentence embedding                    : sentEmbed = [(id, embed/state, lable)]  -> list
    """
    def __init__(self, params):
        self.params     = params
        
    def process(self):
        if self.params.encoder is 'LSTM':
            self.lstm_encoder()
        # if ...
        raise NotImplementedError
        
    def lstm_encoder(self):
        raise NotImplementedError
    
    
