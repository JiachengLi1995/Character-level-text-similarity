from network.data_loader import get_loader
from network.framework import Framework
from network.sentence_encoder import SentenceEncoder
from network.vocab import Vocab
from tqdm import tqdm
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
    ground_truth = {}
    for i in range(len(abbs)):
        if abbs[i] in ground_truth:
            ground_truth[abbs[i]].append(fulls[i])
        else:
            ground_truth[abbs[i]] = []
    
    # indexed_abbs, mask_abb, indexed_target, mask_target = sentence_encoder.tokenize_test("room.leak-shutdown", 'system')
    # input1 = torch.tensor(indexed_abbs).long().unsqueeze(0).cuda()
    # mask1 = torch.tensor(mask_abb).long().unsqueeze(0).cuda()
    # input2 = torch.tensor(indexed_target).long().unsqueeze(0).cuda()
    # mask2 = torch.tensor(mask_target).long().unsqueeze(0).cuda()
    # logits = model(input1, mask1, input2, mask2)
    # print(logits)
    # exit(0)
    abbs_set = list(set(abbs))
    fulls_set = list(set(fulls))
    
    abbs_data = []
    abbs_mask = []
    fulls_data = []
    fulls_mask = []
    for i in range(len(abbs_set)):
        abbs_data_ = []
        abbs_mask_ = []
        fulls_data_ = []
        fulls_mask_ = []
        for j in range(len(fulls_set)):
            indexed_abbs, mask_abb, indexed_target, mask_target = sentence_encoder.tokenize_test(abbs_set[i], fulls_set[j])
            indexed_abbs = torch.tensor(indexed_abbs).long().unsqueeze(0)
            mask_abb = torch.tensor(mask_abb).long().unsqueeze(0)
            indexed_target = torch.tensor(indexed_target).long().unsqueeze(0)
            mask_target = torch.tensor(mask_target).long().unsqueeze(0)

            abbs_data_.append(indexed_abbs)
            abbs_mask_.append(mask_abb)
            fulls_data_.append(indexed_target)
            fulls_mask_.append(mask_target)
        abbs_data.append(abbs_data_)
        abbs_mask.append(abbs_mask_)
        fulls_data.append(fulls_data_)
        fulls_mask.append(fulls_mask_)

    print('Start Evaluating...')
    with torch.no_grad():
        scores = []
        for i in tqdm(range(len(abbs_set))):
            scores_ = []
            for j in range(len(fulls_set)):
                input1, mask1, input2, mask2 = abbs_data[i][j], abbs_mask[i][j], fulls_data[i][j], fulls_mask[i][j]
                if torch.cuda.is_available():
                    input1 = input1.cuda()
                    input2 = input2.cuda()
                    mask1 = mask1.cuda()
                    mask2 = mask2.cuda()
                
                logits = model(input1, mask1, input2, mask2)
                scores_.append(logits.cpu().numpy()[0][0])
            scores.append(scores_)
        scores = np.array(scores)
        print(scores.shape)
    score_arg = np.argsort(scores, axis=1)
    score_sorted = np.sort(scores, axis=1)
    score_arg = score_arg[:,:10]
    score_sorted = score_sorted[:,:10]
    
    result = []
    for i in range(score_arg.shape[0]):
        result.append({'abbreviation': abbs_set[i], 'predicted': list(map(lambda x: fulls_set[x], score_arg[i])), 'ground truth': ground_truth[abbs_set[i]],'score': str(score_sorted[i])})
            
    f = open('result.json','w',encoding='utf8')
    json.dump(result, f)
    f.close()
if __name__ == "__main__":
    main()
    
