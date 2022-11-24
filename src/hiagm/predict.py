#!/usr/bin/env python
# coding:utf-8

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import hiagm.helper.logger as logger
from hiagm.data_modules.collator import Collator
from hiagm.data_modules.dataset import ClassificationDataset
from hiagm.data_modules.preprocess import preprocess_line
from hiagm.data_modules.vocab import Vocab
from hiagm.helper.configure import Configure
from hiagm.models.model import HiAGM


def predict(text, config, model_checkpoint, max_labels=4):
    """
    Predict the labels of a text.
    :param text: The text to be predicted.
    :param config: The configuration of the model.
    :param model_checkpoint: The path of the model checkpoint.
    :param max_labels: The maximum number of labels to be predicted.
    """

    # Clean text 
    data_str = preprocess_line(text)
    
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)

    # build up model and load weights
    checkpoint_model = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TEST')
    hiagm.load_state_dict(checkpoint_model['state_dict'])
    hiagm.to(config.train.device_setting.device)
    # define training objective & optimizer
    collate_fn = Collator(config, corpus_vocab)

    dataset = ClassificationDataset(config, corpus_vocab, corpus_lines=[data_str],mode='TEST')
    data = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    mapper = pd.Series(corpus_vocab.i2v['label'])
    one_example = next(iter(data))
    hiagm.eval()
    out = hiagm(one_example)
    out = torch.sigmoid(out).cpu().tolist()
    out_np = np.array(out)
    out_np = pd.Series(out_np[0,:].flatten())
    df = pd.concat([mapper, out_np], axis=1)
    df.columns = ['label', 'score']
    df_classes = df.sort_values(by='score', ascending=False).iloc[:max_labels]
    print(list(df_classes.label), list(df_classes.score), list(df_classes.index))
    return list(df_classes.label), list(df_classes.score), list(df_classes.index)
    
    


if __name__ == "__main__":
    configs = Configure(config_json_file=sys.argv[1])
    model_checkpoint = sys.argv[2]
    text = sys.argv[3]
    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    logger.Logger(configs)

    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    predict(text, configs, model_checkpoint)
