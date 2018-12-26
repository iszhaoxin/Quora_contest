import os
import sys
import collections

from logging import getLogger

logger = getLogger()

class PostProcess:
    """
    Notes : 
        - If sentence embedding is separable with next layer(Predictor), some extra data augmentation process can adopted here
    Input
        - Sentence embedding
        - Sentence-level dataset
    Output
        - New Sentence embedding
    """
    def __init__(self):
        raise NotImplementedError
        
    def process(self):
        raise NotImplementedError

if __name__ == "__main__":
    pass