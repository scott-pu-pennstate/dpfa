# most part of the code is copy pasted from my another repo:
# https://github.com/scott-pu-pennstate/dktt_light

import os
import math
from collections import defaultdict
import logging
import time

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from dpfa_config import NAME_CONVENTION, VERBOSE

__last_modified__ = time.time() - os.path.getmtime(__file__)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO if VERBOSE == 1 else logging.WARNING)
console = logging.StreamHandler()
console.setLevel(logging.INFO if VERBOSE == 1 else logging.WARNING)
logger.addHandler(console)


class Encoder:
    r"""a simple encoder"""
    def __init__(self, vocab):
        # add special tokens to list
        self.vocab = ['<pad>', '<unk>'] + sorted(vocab)

        # create vocab related dicts
        self.dictionary = defaultdict(lambda: 1)
        self.dictionary.update(dict(
            zip(self.vocab, range(len(self.vocab)))))

    def encode(self, word):
        return self.dictionary[word]


class DataSet:
    def __init__(
            self,
            dataset,
            data_dir,
            cv_idx,
            id_col,
            prob_col,
            score_col,
            time_col,
            max_len,
            **kwargs
    ):
        r"""configure the dataset"""
        self.dataset = dataset
        self.id_col = id_col
        self.prob_col = prob_col
        self.score_col = score_col
        self.time_col = time_col
        self.max_len = max_len
        self.dataset = dataset

        if dataset in ['stat2011', 'assist2017']:
            feature_input = False
        elif dataset in ['nips2020', 'synthetic5']:
            feature_input = True

        # if True, do not need to run
        # 1. encoding
        # 2. self.extract_features function
        self.feature_input = feature_input

        self.train_fpath = os.path.join(
            data_dir,
            NAME_CONVENTION.format(dataset, 'train', cv_idx))
        self.test_fpath = os.path.join(
            data_dir,
            NAME_CONVENTION.format(dataset, 'test', cv_idx))

    def describe_data(self):
        df = pd.concat([self.train_df, self.test_df], axis=0)
        if self.feature_input:
            num_attempts = df[self.prob_col].apply(len).sum()
        else:
            num_attempts = df.shape[0]
        num_students = df[self.id_col].unique().shape[0]

        logger.info(f'dataset = {self.dataset}')
        logger.info(f'num of attempts = {num_attempts / 1000} K')
        logger.info(f'num of students = {num_students}')
        logger.info(f'num of items = {self.item_vocab_size}')

    def load_data(self):
        self.train_df = pd.read_csv(self.train_fpath)
        logger.info(f'train data in {self.train_fpath} is loaded')
        self.test_df = pd.read_csv(self.test_fpath)
        logger.info(f'test data in {self.test_fpath} is loaded')

        if self.feature_input:
            for col in [self.prob_col, self.score_col]:
                for df in [self.train_df, self.test_df]:
                    df[col] = df[col].apply(eval)

        # calculate item vocab size
        df = pd.concat([self.train_df, self.test_df], axis=0)
        if self.feature_input:
            num_items = max(df[self.prob_col].apply(max)) + 1
        else:
            num_items = df[self.prob_col].unique().shape[0] + 2  # pad + unk

        self.item_vocab_size = num_items

    def preprocess(self):
        # data specific cleaning
        if self.dataset == 'stat2011':
            self.train_df[self.time_col] = pd.to_datetime(
                self.train_df[self.time_col])
            self.test_df[self.time_col] = pd.to_datetime(
                self.test_df[self.time_col])

        if not self.feature_input:
            logger.info('input dataset is raw, will perform encode '
                        'and extract features')
            prob_encoder = self.get_encoder()
            logger.info(f'problem encoder is created')

            for df in [self.train_df, self.test_df]:
                # encode
                df[self.prob_col] = df[self.prob_col].apply(prob_encoder.encode)
                df[self.score_col] = df[self.score_col] + 1
            logger.info('train and test data are encoded')

            train_features = self.train_df.groupby(self.id_col).apply(self.extract_features)
            test_features = self.test_df.groupby(self.id_col).apply(self.extract_features)
            logger.info('train and test sequences are extracted')
        else:
            logger.info('input dataset is already a feature dataset'
                        'skipping the encoding and feature extraction step')
            train_features = self.train_df
            test_features = self.test_df

        cols = [self.prob_col, self.score_col]
        dtypes = ['int32', 'int32']
        train_hist_folds, train_next_folds = self.get_folded_seqs(
            train_features[cols].values, dtypes)
        test_hist_folds, test_next_folds = self.get_folded_seqs(
            test_features[cols].values, dtypes)
        logger.info('train and test sequences are folded')

        train_inputs, train_targets = self.get_inputs_and_targets_from_folds(
            train_hist_folds, train_next_folds)
        test_inputs, test_targets = self.get_inputs_and_targets_from_folds(
            test_hist_folds, test_next_folds)
        logger.info('train and test inputs and targets are created')

        return (train_inputs, train_targets), (test_inputs, test_targets)

    def extract_features(self, df):
        df = df.sort_values(self.time_col)

        # flatten df, initialize with id and time_col
        flattened = []
        for col in [self.prob_col, self.score_col]:
            seq = df[col].values.tolist()
            flattened.append((col, [seq]))

        return pd.DataFrame(dict(flattened))

    def get_encoder(self):
        # vocab is based on all data,
        # assuming that item is known to the system before any user data
        prob_vocab = self.get_vocab(
            pd.concat([self.train_df, self.test_df], axis=0), self.prob_col)
        prob_encoder = Encoder(prob_vocab)
        return prob_encoder

    def get_vocab(self, data, col):
        vocab = data[col].unique().tolist()
        return vocab

    def fold_seq(self, seq):
        hist_seqs, next_seqs = [], []
        num_folds = math.ceil(len(seq) / self.max_len)

        for idx in range(num_folds):
            next_start = idx * self.max_len
            next_end = next_start + self.max_len

            hist_start = next_start
            hist_end = min(next_end - 1, len(seq) - 1)

            next_seq = seq[next_start: next_end]
            hist_seq = seq[hist_start: hist_end]

            next_seqs.append(next_seq)
            hist_seqs.append(hist_seq)

        return hist_seqs, next_seqs

    def get_folded_seqs(self, df, dtypes):
        hist_folds = [[] for _ in range(df.shape[1])]
        next_folds = [[] for _ in range(df.shape[1])]

        for row in tqdm(range(df.shape[0])):
            for idx, seq in enumerate(df[row]):
                col_hist_seq, col_next_seq = self.fold_seq(seq)
                hist_folds[idx].extend(col_hist_seq)
                next_folds[idx].extend(col_next_seq)

        hist_padded_folds = [
            tf.keras.preprocessing.sequence.pad_sequences(folds, maxlen=self.max_len, dtype=dtype) for folds, dtype in zip(hist_folds, dtypes)]
        next_padded_folds = [
            tf.keras.preprocessing.sequence.pad_sequences(folds, maxlen=self.max_len, dtype=dtype) for folds, dtype in zip(next_folds, dtypes)]
        return hist_padded_folds, next_padded_folds

    @staticmethod
    def get_inputs_and_targets_from_folds(hist_folds, next_folds):
        inputs = {
            'history_items': hist_folds[0],
            'history_corrects': hist_folds[1],
            'next_items': next_folds[0]}

        targets = next_folds[1]

        return inputs, targets