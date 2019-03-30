#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data.py
# @Author: Betafringe
# @Date  : 2019-03-29
# @Desc  : 
# @Contact : betafringe@foxmail.com
import os
import numpy as np
import json

class DataReader(object):
    def __init__(self, wordemb_path,
                 postag_dict_path,
                 label_dict_path,
                 train_data_path,
                 test_data_path,
                 savepath):
        self._wordemb_path = wordemb_path
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._savepath = savepath

    def read_embedding_file(self):
        with open(self._wordemb_path, 'r', encoding='utf-8') as fr:
            file = fr.readlines()
        return file

    def read_test_data(self):
        with open(self._test_data_path, 'r', encoding='utf-8') as fr:
            file = json.loads()

    def read_train_data(self):
        with open(self._train_data_path, 'r', encoding='utf-8') as fr:
            file = fr.readlines()
        return file

    def get_vacab(self, from_embedding=True):
        vocab = set()
        if from_embedding:
            vocabfile = self.read_embedding_file()
            for line in vocabfile:
                word = line.rstrip().split()[0]
                vocab.add(word)
        else:
            vocabfile = self.read_train_data()
            for line in vocabfile:
                words = line.split(' ')
                for word in words:
                    vocab.add(word)
            print('--adding unk word--')
            vocab.add('unk')
        return vocab

    def get_dict(self, padding=True):
        vocab = self.get_vacab()
        if padding:
            ind2w = {i + 1: w for i, w in enumerate(vocab)}
        else:
            ind2w = {i: w for i, w in enumerate(vocab)}
        w2ind = {w: i for i, w in ind2w.items()}
        return ind2w, w2ind

    def get_embedding_dict(self):
        w2v_file = self.read_embedding_file()[1:]
        w2ind = self.get_dict(padding=True)[-1]
        w2v = {}
        id2v = {}
        for line in w2v_file:
            line = line.strip().split()
            vec = np.array(line[1:301]).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            w2v[line[0]] = vec

        for w in w2ind:
            if(w in w2v.keys()):
                id2v[w2ind[w]] = w2v[w]
            else:
                vec = np.random.randn(300)
                vec = vec / (np.linalg.norm(vec) + 1e-6)
                id2v[w2ind[w]] = vec
        return w2v, id2v

    def extrac_wvs(self):
        W = []
        w2v, id2v = self.get_embedding_dict()
        for idx in id2v:
            W.append(id2v[idx])
        return W

class DataGenerater(object):
    def __init__(self, batchsize, train, test):
        self._batchsize = batchsize
        self._training = train
        self._testing = test

    def datagenerate(self):
        pass


if __name__ == '__main__':
    # initialize data generator
    data_generator = DataReader(
        wordemb_path='../../data/dict/word_idx',
        postag_dict_path='./dict/postag_dict',
        label_dict_path='../dict/p_eng',
        train_data_path='../../data/seg_txt.txt',
        test_data_path='.../../data/dev_data.json')
    test = data_generator.get_vacab()
    print(len(test))
    # # prepare data reader
    # ttt = data_generator.get_test_reader()
    # for index, features in enumerate(ttt()):
    #     input_sent, word_idx_list, postag_list, label_list = features
    #     print input_sent.encode('utf-8')
    #     print '1st features:', len(word_idx_list), word_idx_list
    #     print '2nd features:', len(postag_list), postag_list
    #     print '3rd features:', len(label_list), '\t', label_list