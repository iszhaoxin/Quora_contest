from __future__ import absolute_import

import os, sys, time
import io, pickle
import re, random
import argparse, inspect
import subprocess

import numpy as np
import torch

from .logger import create_logger
from torch import optim
from logging import getLogger

MAIN_DUMP_PATH = "../data/export"
if not os.path.exists(MAIN_DUMP_PATH): os.mkdir(MAIN_DUMP_PATH)

logger = getLogger()


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")

def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        # chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        timestamp = time.strftime('%b%d_%Hh%Mm%Ss')
        while True:
            # exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, timestamp)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path

def export_embeddings(emb, params):
    """
    Export embeddings to a text or a PyTorch file.
    """
    assert params.export in ["txt", "pth"]

    # text file
    if params.export == "txt":
        path = os.path.join(params.exp_path, 'vectors-%s.txt' % emb)
        # source embeddings
        logger.info('Writing source embeddings to %s ...' % path)
        with io.open(path, 'w', encoding='utf-8') as f:
            raise NotImplementedError
            
    # PyTorch file
    if params.export == "pth":
        path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.src_lang)
        logger.info('Writing source embeddings to %s ...' % path)
        torch.save({'dico': params.src_dico, 'vectors': emb}, path)
        
def export_weight(params, model):
    path = os.path.join(params.exp_path, 'model.weight')
    raise NotImplementedError
    