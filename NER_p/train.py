#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: Betafringe
# @Date  : 2019-03-29
# @Desc  : 
# @Contact : betafringe@foxmail.com
import argparse
import sys
from datetime import time




def main(args):
    args, model, optimizer, params, dicts = init(args)
    model = model.cuda()
    epochs_trained = train_epoches(args, model, optimizer, params, dicts)


def init(args):

    return args, model, optimizer, params, dicts


def train_epoches(args, model, optimizer, params, dicts):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train on relation extraction')
    parser.add_argument("--data_path", type=str, required=False, default="../data/seg_text_comma.txt",
                        help="path to a file containing sorted train data")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("NER_p", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "saved"], help="NER_p")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)", dest='cell_type',
                        default='gru')
    parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
                        help="optional flag for rnn to use a bidirectional NER_p")
    parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=4,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, "
                             "give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--pool", choices=['max', 'avg'], required=False, dest="pool", help="which type of pooling to do (logreg NER_p only)")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of NER_p weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--test-NER_p", type=str, dest="test_model", required=False, help="path to a saved NER_p to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)