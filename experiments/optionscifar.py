#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=300, #400, #300,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,#2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=25, #50,
                        help="local batch size: B")
    parser.add_argument('--local_sgd_it', type=int, default=1,
                        help="Number of local SGD iterations (client training size: local_bs * local_sgd_iter)")
    parser.add_argument('--mia_sample_num', type=int, default=5000, #6000,
                        help="Number of samples to train MIA attacker model")
    parser.add_argument('--attacker_model_ep', type=int, default=50,
                        help="Number of training epochs of MIA model")
    parser.add_argument('--lr', type=float, default=0.1, #0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--attacker_model_reg', type=str, default='l2', #l2
                    help="L1 or L2 regularizer")
    parser.add_argument('--penality', type=float, default=0.0,
                help="L1 or L2 regularizer")
    
    parser.add_argument('--prop', type=float, default=0.1,
            help="Proportion of attackers in case of poisoning attacks or honest clients with the target record in case of membership attacks")
    
    parser.add_argument('--alpha', type=float, default=0.4,
            help="Tradeoff parameter")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #device= "cpu"
    
    parser.add_argument('--device', default=str(device), help="To use cuda, set \
                        to a specific GPU ID. To use Apple M1. use meta, Default set to use CPU.")
    
    #parser.add_argument('--device', default="cpu", help="To use cuda, set \
    #                to a specific GPU ID. To use Apple M1. use meta, Default set to use CPU.")
    
    
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,#1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
