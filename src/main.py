import os, sys
import time
import json
import argparse
import numpy as np
import torch
from collections import OrderedDict

from models.model import build_model
from models.trainer import Trainer
from models.evaluator import Evaluator


from utils.utils import bool_flag, initialize_exp

__version__ = '0.1'

with open('utils/path.json','r') as f:
    paths = json.load(f)

# main
parser = argparse.ArgumentParser(description='Model')

# Read data
parser.add_argument("--seed",       type=int,   default=-1,     help="Initialization seed")
parser.add_argument("--paths",      type=dict,  default=paths,  help="Locations of folder")

# Write and Log
parser.add_argument("--export",     type=bool_flag, default=False,  help="Export embeddings or not")
parser.add_argument("--verbose",    type=int,       default=2,      help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--logging",    type=bool_flag, default=True,   help="Whether to draw result for evaluation")
parser.add_argument("--exp_path",   type=str,       default="",     help="Location to store experiment logs and models")
parser.add_argument("--exp_name",   type=str,       default="baseline",     help="Experiment name")
parser.add_argument("--exp_id",     type=str,       default="",     help="Experiment ID")


# Pre-processing
## Extra information
## Data augumentation
parser.add_argument("--input_dropout",  type=float,     default=0.,    help="input dropout")
parser.add_argument("--data_noising",   type=bool_flag, default=False, help="Refer paper : Data noising as smoothing in neural network language models")


# Word Embedding
parser.add_argument("--PCA",        type=bool_flag, default=True, help="Embedding dimension")
parser.add_argument("--word2vec",   type=bool_flag, default=True,  help="Reload pre-embeddings word2vec")
parser.add_argument("--glove",      type=bool_flag, default=False, help="Reload pre-embeddings glove")
parser.add_argument("--fasttext",   type=bool_flag, default=False, help="Reload pre-embeddings fasttext")
parser.add_argument("--paragram",   type=bool_flag, default=False, help="Reload pre-embeddings paragram")
parser.add_argument("--concat",     type=bool_flag, default=False, help="Concat serveral embeddings")
parser.add_argument("--label",      type=bool_flag, default=False, help="Joint Embedding of Words and Labels for Text Classification")
parser.add_argument("--padding",    type=bool_flag, default=False, help="Pad sentences to fix length")
parser.add_argument("--add_POS",    type=bool_flag, default=False, help="Add part-of-speech embedding to word embedding")
parser.add_argument("--add_position",   type=bool_flag, default=False, help="Add position embedding to word embedding")
parser.add_argument("--add_dependency", type=bool_flag, default=False, help="Add dependency embedding to word embedding")

# Post-processing
parser.add_argument("--postprocess",  type=bool_flag, default=True,  help="Whether to take embedding as parameters")
parser.add_argument("--trainable",  type=bool_flag, default=True,  help="Whether to take embedding as parameters")
parser.add_argument("--whitening",          type=bool_flag, default=True,   help="")
parser.add_argument("--add_diversity",      type=bool_flag, default=True,   help="Caculate distances between words in sentence")
parser.add_argument("--normalization",      type=bool_flag, default=True,   help="")
parser.add_argument("--add_other_features", type=bool_flag, default=True,   help="")

# Model about Encoder
parser.add_argument("--encoder",    type=str, default='SIF', help="The model for encoder")

## LSTM 
parser.add_argument("--lstm_in_dim",        type=int,       default=2048,   help="")
parser.add_argument("--lstm_hid_dim",       type=int,       default=2048,   help="")
parser.add_argument("--lstm_layers",        type=int,       default=2,      help="")
parser.add_argument("--lstm_bias",          type=float,     default=0,      help="")
parser.add_argument("--lstm_dropout",       type=float,     default=0.,     help="")
parser.add_argument("--lstm_bidirectional", type=bool_flag, default=False,  help="")
## RNN , Bi-LSTM , Att-LSTM, SIF, Power_mean , GEM, CNN , Att-CNN , Att-LSTM

# Prediction model
parser.add_argument("--predictor",      type=str, default='logistic', help="")

# Trainer
## Basic parameter settings
parser.add_argument("--cuda",       type=bool_flag, default=False,  help="")
parser.add_argument("--n_epochs",   type=int,       default=300,    help="")
parser.add_argument("--epoch_size", type=int,       default=1000000,help="")
parser.add_argument("--batch_size", type=int,       default=32,     help="")
parser.add_argument("--lr",         type=float,     default=0.01,   help="Learning rate")
## optimizer
parser.add_argument("--optimizer",  type=str,       default="sgd,lr=0.01",   help="")
## lr decay
parser.add_argument("--decay",      type=bool_flag, default=False,  help="Decay the learning rate")
parser.add_argument("--lr_decay",   type=float,     default=0.98,   help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr",     type=float,     default=1e-6,   help="Minimum learning rate (SGD only)")
## Fine-tuning tircks
parser.add_argument("--tricks",     type=float,     default=1e-6,   help="")

# Evalutor
parser.add_argument("--Draw_result",type=bool_flag,   default=True,  help="Whether to draw result for evaluation")
parser.add_argument("--Accuracy",   type=bool_flag,   default=False, help="Accuracy")
parser.add_argument("--Recall",     type=bool_flag,   default=False, help="Recall")
parser.add_argument("--F1",         type=bool_flag,   default=False, help="F1")
parser.add_argument("--Precision",  type=bool_flag,   default=False, help="Precision")
parser.add_argument("--ROC",        type=bool_flag,   default=False, help="Accuracy")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.input_dropout < 1
# assert os.path.isfile(params.paths['word2vec_path'])
# assert os.path.isfile(params.paths['glove_path'])
# assert os.path.isfile(params.paths['fasttext_path'])
# assert os.path.isfile(params.paths['paragram_path'])


logger = initialize_exp(params)

# ------- bulid model -------
data, model = build_model(params, True)
train_sents, train_labels, val_sents, val_labels = data
# ------- trainer settings -------
trainer = Trainer(params, train_sents, train_labels , model)
trainer.train()
# # ------- evaluator -------
evaluator = Evaluator(params, val_sents, val_labels, model)
# # ------- Export model and embeddings -------
# if params.export:
#     trainer.reload_best()
#     trainer.export()
