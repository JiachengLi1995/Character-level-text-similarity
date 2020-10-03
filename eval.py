from network.data_loader import get_loader
from network.framework import Framework
from network.sentence_encoder import SentenceEncoder
from network.vocab import Vocab
import models
from models.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)

def __load_data__(name, root):
    path = os.path.join(root, name + ".json")
    if not os.path.exists(path):
        print("[ERROR] Data file does not exist!")
        assert(0)
    data = json.load(open(path))
    abbs = []
    fulls = []
    for line in data:

        abbs.append(line['abbreviation'])
        fulls.append(line['full-text'])
    return abbs, fulls



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--test', default='test_sim',
            help='test file')
    parser.add_argument('--N', default=2, type=int,
            help='Number of example concencate')
    parser.add_argument('--batch_size', default=1, type=int,
            help='batch size')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='cnn, lstm')
    parser.add_argument('--max_length', default=32, type=int,
           help='max length')
    parser.add_argument('--hidden_size', default=64, type=int,
           help='hidden size')
    
    opt = parser.parse_args()
    N = opt.N

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


    prefix = '-'.join([model_name, encoder_name, opt.train, 'dev', str(N)])
    model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    state_dict = __load_model__(ckpt)['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    

    abbs, fulls = __load_data__(opt.test, './data')
    
    abbs_data = []
    abbs_mask = []
    fulls_data = []
    fulls_mask = []
    for i in range(len(abbs)):
        indexed_abbs, mask_abb, indexed_target, mask_target = sentence_encoder.tokenize_test(abbs[i], fulls[i])
        indexed_abbs = torch.tensor(indexed_abbs).long().unsqueeze(0)
        mask_abb = torch.tensor(mask_abb).long().unsqueeze(0)
        indexed_target = torch.tensor(indexed_target).long().unsqueeze(0)
        mask_target = torch.tensor(mask_target).long().unsqueeze(0)

        abbs_data.append(indexed_abbs)
        abbs_mask.append(mask_abb)
        fulls_data.append(indexed_target)
        fulls_mask.append(mask_target)
    

    iter_right = 0.0
    iter_sample = 0.0
    loss_list = []
    with torch.no_grad():
        for it in range(len(abbs_data)):
            input1, mask1, input2, mask2 = abbs_data[it], abbs_mask[it], fulls_data[it], fulls_mask[it]
            if torch.cuda.is_available():
                input1 = input1.cuda()
                input2 = input2.cuda()
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()
                
            #print(input1, input2)
            logits = model(input1, mask1, input2, mask2)
            right = int(logits.cpu().numpy()[0][0] < 0.5)
            # if right==1:
            #     print(abbs[it],fulls[it])
            iter_right += right
            iter_sample += 1
            # if iter_sample>100:
            #     exit(0)
            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:5.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()


if __name__ == "__main__":
    main()
    
