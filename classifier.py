from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sentence_embedding import SimpleEmbedding

class LRSentenceClassifier:
    def __init__(self, sentence_embedding, a=0.001):
        self.embed = sentence_embedding

        self.lr = LogisticRegression(
                                    penalty="l2", dual=False, tol=0.0001, 
                                    C=1.0, fit_intercept=True, intercept_scaling=1, 
                                    class_weight=None, random_state=None, 
                                    solver="lbfgs", max_iter=100, multi_class="warn", 
                                    verbose=0, warm_start=False, n_jobs=None)

    def train(self, sentences, labels):
        sentences = self.embed.calc_embeddings(sentences)
        self.lr.fit(sentences, labels)

    def predict(self, sentences):
        sentences = self.embed.calc_embeddings(sentences)
        return self.lr.predict(sentences)

    def score(self, sentences, labels):
        pred = self.predict(sentences)
        print("acc:", accuracy_score(labels, pred))
        print("precision:", precision_score(labels, pred))
        print("recall:", recall_score(labels, pred))
        print("conf_matrix:", confusion_matrix(labels, pred), sep="\n")


if __name__ == "__main__":
    import pandas as pd
    from gensim.models import KeyedVectors
    from make_vocab import make_vocab
    import argparse
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-train', type=str, default='train.tokenized')
    parser.add_argument('-val', type=str, default='val.tokenized')
    parser.add_argument('-embed', type=str, default='GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')

    args = parser.parse_args()

    word2index, vectors = make_vocab(args.train, args.embed, vocab_size=None)
    
    se = SimpleEmbedding(args.train, vectors, word2index, a=0.001)

    train_df = pd.read_csv(args.train)
    train_sents = [sent.split(" ") for sent in train_df["question_text"].values]
    train_labels = train_df["target"].values

    val_df = pd.read_csv(args.train)
    val_sents = [sent.split(" ") for sent in train_df["question_text"].values]
    val_labels = train_df["target"].values
    
    
    lr = LRSentenceClassifier(se)
    lr.train(train_sents, train_labels)
    lr.score(val_sents, val_labels)