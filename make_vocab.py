from torchtext import data, datasets
from gensim.models import KeyedVectors
import argparse
import torch
import numpy as np

def load_embedding(path):
    """
    input: embedding path
    output: word2index dict, vectors array float32
    """
    # build Vocab
    if "GoogleNews-vectors-negative300.bin" in path:
        embed = KeyedVectors.load_word2vec_format(path, binary=True)
        vectors = embed.vectors
        word2index = {w:i for i, w in enumerate(embed.index2word)}
        return word2index, vectors
    
    f = open(path)
    dim = None

    if "wiki-news-300d-1M.vec" in path:
        dim = int(f.readline().split(" ")[1])
    elif "glove.840B.300d.txt" in path or "paragram_300_sl999.txt" in path:
        pass
    else:
        raise ValueError

    vectors = list()
    word2index = dict()
    for line in f:
        tokens = line.split(" ")
        if dim is None:
            dim = len(tokens[1:])    
        word = tokens[:-1*dim][0]
        vector = [float(num) for num in tokens[-1*dim:]]
        vectors.append(vector)
        word2index[word] = len(word2index)
    
    f.close()
    
    return word2index, np.array(vectors,dtype=np.float32)

def make_vocab(source_path, embed_path, vocab_size):
    QID = data.Field(sequential=False, use_vocab=False)
    TEXT = data.Field(sequential=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    pos = data.TabularDataset(
            path=source_path, format='csv',  
            fields=[('qid', QID), 
                    ('text', TEXT),
                    ('label', LABEL)],
            skip_header=True)

    w2i, vec = load_embedding(embed_path)
    vec = torch.FloatTensor(vec)

    TEXT.build_vocab(
            pos,
            max_size=vocab_size, min_freq=1, vectors=None,
            unk_init=None, vectors_cache=None, specials_first=True)
    vocab = TEXT.vocab
    vocab.set_vectors(stoi=w2i, vectors=vec, dim=vec.shape[1])
    
    return vocab.stoi, vocab.vectors.numpy()
    

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-source', type=str, default='dev.csv')     # train.csv
    parser.add_argument('-embed', type=str, default='glove.840B.300d/glove.840B.300d.txt')
    parser.add_argument('-vocab_size', type=int, default=50000)

    args = parser.parse_args()

    word2index, vectors = make_vocab(args.source, args.embed, args.vocab_size)
    