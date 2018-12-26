from collections import Counter
import csv
import numpy as np

class SimpleEmbedding:
    def __init__(self, corpus_path, word_vectors, word2index, a=0.001):
        self.word_vectors = word_vectors
        self.w2i = word2index
        self.a = a
        self.word_prob = self.calc_word_prob(corpus_path)

    def get_word_embed(self, word):
        return self.word_vectors[self.w2i[word]]

    def calc_word_prob(self, path):
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
        return {word: num / doc_size for word, num in cnt.items()}


    def calc_embeddings(self, sentences):
        """ 
        input: tokenized sources [s1, s2, ...]
        """
        sent_embeddings = list()
        for sent in sentences:
            sent_size = 1 / len(sent)
            vs = sum(self.a / (self.a + self.word_prob[word]) * self.get_word_embed(word)  for word in sent) / sent_size
            sent_embeddings.append(vs)
        sent_embeddings = np.array(sent_embeddings, dtype=np.float32)
        u, _, _ = np.linalg.svd(sent_embeddings.T)
        sent_embeddings = np.array([vs - u @ u.T @ vs for vs in sent_embeddings], dtype=np.float32)
        return sent_embeddings



