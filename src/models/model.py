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
    dataset   = Dataset()
    train     = dataset.process(params.train_path)
    valid     = dataset.process(params.valid_path)
    sentenceD = PreProcess(params, train)
        
    # load embeddings
    logger.info('----> load embeddings <----\n\n')
    wordEmbedding   = WordEmbedding(params)
    
    # Post-processing
    if params.postprocess:
        logger.info('----> Post-processing <----\n\n')
        wordEmbedding    = PreProcess(params, train)

    # Sentence embedding (Encoder) and predictor
    logger.info('----> Sentence embedding and predictor <----\n\n')
    encoder         = Encoder()
    predictor       = Predictor()

    model           = predictor(encoder)
    
    if params.cuda:
        wordEmbedding.cuda()
        if with_dis:
            model.cuda()

    return wordEmbedding, model

