import csv
import spacy
import random
import numpy as np
from collections import Counter, defaultdict
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
    def __init__(self, params, wordEmbedding):
        self.params     = params
        self.wordEmbedding = wordEmbedding
        
        if self.params.encoder is 'LSTM':
            self.encoder = self.lstm_encoder()
        elif self.params.encoder is 'SIF':
            self.encoder = self.sif_encoder()
        else:
            raise ValueError

        self.calc_embeddings = self.encoder.calc_embeddings
        
    def lstm_encoder(self):
        raise NotImplementedError

    def sif_encoder(self):
        encoder = SimpleEmbedding(
            corpus_path = self.params.paths['train_path'],
            word2index  = self.wordEmbedding.vocab.stoi,
            vectors     = self.wordEmbedding.vocab.vectors
        )
        return encoder

class SimpleEmbedding:
    
    __name__ = "SIF"

    def __init__(self, corpus_path, vectors, word2index, a=0.001):
        self.vectors = vectors
        self.w2i = word2index
        self.a = a
        self.word_prob = self._calc_word_prob(corpus_path)

    def _get_word_embed(self, word):
        return self.vectors[self.w2i[word]].numpy()

    def _calc_word_prob(self, path):
        """
        input: tokenized csv path
        """
        cnt = Counter()
        with open(path) as f:
            reader = csv.reader(f, delimiter=",", doublequote=True,
                                lineterminator="\n", quotechar='"')
            header = next(reader)
            doc_size = 0
            for row in reader:
                tokens = row[1].split(" ")
                cnt.update(tokens)
                doc_size += len(tokens)
        return defaultdict(int, {word: num / doc_size for word, num in cnt.items()})


    def calc_embeddings(self, sentences):
        """ 
        input: tokenized sources [s1, s2, ...]
        """
        sent_embeddings = list()
        for sent in sentences:
            sent_size = 1 / len(sent)
            for word in sent:
                word_embedding = self._get_word_embed(word)
            vs = sum([self.a / (self.a + self.word_prob[word]) * self._get_word_embed(word)  for word in sent]) / sent_size
            sent_embeddings.append(vs)
        sent_embeddings = np.array(sent_embeddings, dtype=np.float32)
        u, _, _ = np.linalg.svd(sent_embeddings.T)
        sent_embeddings = np.array([vs - u @ u.T @ vs for vs in sent_embeddings], dtype=np.float32)
        return sent_embeddings



