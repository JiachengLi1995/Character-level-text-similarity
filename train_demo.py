from network.data_loader import get_loader
from network.framework import Framework
from network.sentence_encoder import SentenceEncoder
from network.vocab import Vocab
import models
from models.proto import Proto
from models.siamese import Siamese
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--val', default='dev',
            help='val file')
    parser.add_argument('--test', default='test',
            help='test file')
    parser.add_argument('--N', default=2, type=int,
            help='Number of example concencate')
    parser.add_argument('--batch_size', default=32, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='cnn, lstm')
    parser.add_argument('--max_length', default=32, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-5, type=float,
           help='learning rate')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam')
    parser.add_argument('--hidden_size', default=64, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--only_test', action='store_true',
           help='only test')

    opt = parser.parse_args()
    N = opt.N
    
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    print('Preparing vocab.')

    vocab = Vocab(opt.train, './data')
    char2id = vocab.prepare_vocab()
    print('Prepare done!')
    sentence_encoder = SentenceEncoder(char2id, max_length, opt.hidden_size, opt.hidden_size, encoder = opt.encoder)
   
    
    train_data_loader = get_loader(opt.train, sentence_encoder, N=N, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder, N=N, batch_size=batch_size)
    test_data_loader = get_loader(opt.test, sentence_encoder, N=N, batch_size=batch_size)
        
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    else:
        raise NotImplementedError
    

    framework = Framework(train_data_loader, val_data_loader, test_data_loader)
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N)])
    
    model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

  
        framework.train(model, prefix, batch_size,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, train_iter=opt.train_iter, val_iter=opt.val_iter)
    else:
        ckpt = opt.load_ckpt

    acc = framework.eval(model, batch_size, opt.test_iter, ckpt=ckpt)
    print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()
