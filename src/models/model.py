import sys
sys.path.append("..")
from modules.pre_process import Dataset, PreProcess
from modules.word_embedding import WordEmbedding
from modules.post_process import PostProcess
from modules.encoder import Encoder
from modules.predictor import Predictor

import logging

logger = logging.getLogger()

# build model / trainer / evaluator

def build_model(params, with_dis):
    """
    Two tasks
    - Process text data to word embedding and statistical information used as features
    - Build all components of the model.
    """
    # Pre-processing
    logger.info('----> Pre-processing <----\n\n')
    dataset   = Dataset(params)
    # train     = dataset.process(params.paths['train_path'])
    # valid     = dataset.process(params.paths['valid_path'])
    # sentenceD = PreProcess(params, train)
    train_sents, train_labels = dataset.train_set()
    val_sents, val_labels = dataset.valid_set()
    data = (train_sents, train_labels, val_sents, val_labels)
    
    # # load embeddings
    logger.info('----> load embeddings <----\n\n')
    embedTable   = WordEmbedding(params)
    
    # # Post-processing
    # if params.postprocess:
    #     logger.info('----> Post-processing start<----\n\n')
    #     wordEmbedding    = PostProcess(params, train)

    # Sentence embedding (Encoder) and predictor
    logger.info('----> Sentence embedding and predictor <----\n\n')
    encoder         = Encoder(params, embedTable)
    predictor       = Predictor(params, encoder)

    # if params.cuda:
    #     wordEmbedding.cuda()
    #     if with_dis:
    #         model.cuda()

    return data, predictor
