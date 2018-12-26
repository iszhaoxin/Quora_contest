import numpy
import spacy
import random
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, classification_report


class Predictor:
    """
    Notes : 
        - Classification model 
    Input : 
        - Word embedding                    : sentEmbed = [(id, embedlist, lable)]  -> list
        - 
    """
    def __init__(self, params, encoder):
        self.params     = params
        self.encoder    = encoder
        
        if self.params.predictor=="logistic":
            self.predictor = self._logistic()
            
    def _logistic(self):
        lr = LRSentenceClassifier(self.encoder)
        return lr
        # lr.train(train_sents, train_labels)
        # lr.score(val_sents, val_labels)

class LRSentenceClassifier:
    def __init__(self, encoder, a=0.001):
        self.encoder = encoder

        self.lr = LogisticRegression(
                                    penalty="l2", dual=False, tol=0.0001, 
                                    C=1.0, fit_intercept=True, intercept_scaling=1, 
                                    class_weight=None, random_state=None, 
                                    solver="lbfgs", max_iter=100, multi_class="warn", 
                                    verbose=0, warm_start=False, n_jobs=None)

    def train(self, sentences, labels):
        sentences = self.encoder.calc_embeddings(sentences)
        self.lr.fit(sentences, labels)

    def _predict(self, sentences):
        sentences = self.encoder.calc_embeddings(sentences)
        return self.lr.predict(sentences)

    def score(self, sentences, labels):
        pred = self.predict(sentences)
        report = classification_report(labels, pred)
        print("report:", report)
        print("acc:", accuracy_score(labels, pred))
        print("precision:", precision_score(labels, pred))
        print("recall:", recall_score(labels, pred))
        print("conf_matrix:", confusion_matrix(labels, pred), sep="\n")

