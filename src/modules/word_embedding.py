import numpy as np
import spacy
import random
import torch
from torch import nn
from spacy.language import Language
from torchtext import data, datasets
from gensim.models import KeyedVectors

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
        # Size of XX types
        POSTYPES = 30 
        MAXLEN = 50
        DEPTYPES = 30
        # Dimension of pos embedding
        POSDIM = 10
        POSITIONDIM = 10
        DEPDIM = 10

        # final dimension after concatenation
        if self.params.add_POS:
            self.posEmbedding = torch.randn(POSTYPES, POSDIM, dtype=torch.float32)
        if self.params.add_position:
            self.postionEembedding = torch.randn(MAXLEN, POSITIONDIM, dtype=torch.float32)
        if self.params.add_dependency:
            self.depEmbedding = torch.randn(DEPTYPES, DEPDIM, dtype=torch.float32)        

        # word2vec : vocab.stoi     vectors: vocab.vectors
        self.dims, self.vocab = self._load_embedding()

        # Use for embedding lookup
        self.embed = nn.Embedding(len(self.vocab), self.dims)
        self.embed.weight.data.copy_(self.vocab.vectors)
        
    def lookup(self, sentences):
        tokenize = (lambda s: s.split())
        indexs = []
        for sent in sentences:
            words = tokenize(sent)
            index = [self.vocab.stoi[word] for word in words]
            indexs.append(index)
        lookup_tensor = torch.tensor(indexs, dtype=torch.float32)
        embeddings = self.embeds(lookup_tensor)
        
    def context_embeddings(self, sentences):
        """
        Input: 
            - sentences: sequences of text
            - sent_index: corresponding index sequences
        Output:
            - wordEmbed_list: lists of contextualized word embedding (3-D:(1:sent_index, 2:word_index, 3:context_embedding))
        """

        # TODO: lookup word embeddings in original embedding vectors
        wordEmbeds = self.lookup(sentences)
        
        sentIndex = self._get_index(sentences)

        if self.params.add_POS:
            pos_tag = self._get_dep_tag(sentences)
            wordEmbeds = self._set_pos_embed(wordEmbeds, sentIndex, pos_tag)
        if self.params.add_position:
            position_tag = self._get_position_tag(sentences)
            wordEmbeds = self._set_position_embed(wordEmbeds, sentIndex, position_tag)
        if self.params.add_dependency:
            dep_tag = self._get_dep_tag(sentences)
            wordEmbeds = self._set_dep_embed(wordEmbeds, sentIndex, dep_tag)

        return wordEmbeds
            
    def _load_embedding(self):
        """
        all_vectors : list[list[int, dict, numpy.adarray]]
            - int : dimension
            - vocab :
                - stoi : word2index(dict)
                - vectors : word embeddings(numpy.adarray)
        """
        all_vectors = []
        if self.params.word2vec:
            all_vectors.append(self._load_word2vec())
        if self.params.glove:
            all_vectors.append(self._load_glove())
        if self.params.fasttext:
            all_vectors.append(self._load_fasttext())
        if self.params.paragram:
            all_vectors.append(self._load_paragram())
        
        assert((len(all_vectors)==1) != self.params.concat)
        if len(all_vectors) == 1:
            dim, word2index, vectors = all_vectors[-1]
        if self.params.concat:
            dim, word2index, vectors = self._concat(all_vectors)
        
        vocab = self._make_vocab(self.params.paths['train_path'], word2index, vectors)

        return dim, vocab

    def _concat(self, all_vectors):
        """
        Concatenate several word embeddings into long embeddings, and generate new word2index and dim
        """
        # Incorprate all words into a new index2word
        wordSet = set()
        concat_dim = 0
        for dim, word2index, _ in all_vectors:
            wordSet |= set(word2index.keys())
            concat_dim += dim
        concat_word2index = {w:i for i, w in enumerate(wordSet)}
        
        # initialize new vectors
        concat_vectors = np.zeros(len(wordSet), concat_dim, dtype=np.float32)
        
        # Process unseen words in each vectors & Make new concat vectors
        for word in concat_word2index:
            index  = concat_word2index[word]
            concat_vector = []
            for dim, word2index, vectors in all_vectors:
                if word in word2index:
                    concat_vector.append(vectors[word2index[word]])
                else:
                    concat_vector.append(np.random.randn(dim)*0.06)
            concat_vectors[index] = np.hstack(concat_vector)
        return concat_dim, concat_word2index, concat_vectors    

    def _make_vocab(self, source_path, word2index, vectors):
        """
        Build sub-vocab for specifical corpus from word embedding dataset (like word2vec)
        """

        QID = data.Field(sequential=False, use_vocab=False)
        if self.params.padding:
            TEXT = data.Field(sequential=True, fix_length=MAXLEN, pad_first=True)
        else:
            TEXT = data.Field(sequential=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        
        pos = data.TabularDataset(
                path=source_path, format='csv',  
                fields=[('qid', QID), 
                        ('text', TEXT),
                        ('label', LABEL)],
                skip_header=True)

        TEXT.build_vocab(
                pos,
                max_size=vectors.shape[0], min_freq=1, vectors=None,
                unk_init=None, vectors_cache=None, specials_first=True)
        vocab = TEXT.vocab
        vocab.set_vectors(stoi=word2index, vectors=torch.Tensor(vectors), dim=vectors.shape[1])        
        # stoi: A collections.defaultdict instance mapping token strings to numerical identifiers.
        return vocab
        
    def _load_word2vec(self):
        """
        Load word2vec from params.word
        """
        embed = KeyedVectors.load_word2vec_format(self.params.paths['word2vec_path'], binary=True)
        vectors = embed.vectors
        word2index = {w:i for i, w in enumerate(embed.index2word)}
        dim = 300
        return dim, word2index, vectors
    def _load_glove(self):
        """
        Load glove from params.word
        """
        raise NotImplementedError
    def _load_fasttext(self):
        """
        Load fasttext from params.word
        """
        raise NotImplementedError
    def _load_paragram(self):
        """
        Load paragram from params.word
        """
        f = open(path)
        dim = int(f.readline().split(" ")[1])

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
        
        return dim, word2index, np.array(vectors,dtype=np.float32)
            
    def _get_pos_tag(self, sentences):
        raise NotImplementedError

    def _get_position_tag(self, sentences):
        raise NotImplementedError

    def _get_dep_tag(self, sentences):
        raise NotImplementedError

    def _set_pos_embed(self, sentences):
        raise NotImplementedError

    def _set_position_embed(self, sentences):
        raise NotImplementedError
    
    def _set_dep_embed(self, sentences):
        raise NotImplementedError

