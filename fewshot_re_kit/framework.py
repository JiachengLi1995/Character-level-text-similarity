import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
import random
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.mse = nn.MSELoss()
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        
        label = label.float()
        tmp= label * logits**2
        maxi = torch.max((1 - logits),other=torch.zeros_like(logits))
        tmp2 = (1-label) * maxi**2
        batch_size = label.size(0)
        return torch.sum(tmp + tmp2)/batch_size/2
    def mse_loss(self, logtis, label):
        label = label.float()
        return self.mse(logtis, label)

    def accuracy(self, logits, label):
        predicted = (logits<0.5).long()
        # print(logits)
        # print(predicted)
        # exit(0)
        return torch.mean((predicted == label).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
    
    def __load_model__(self, ckpt):
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
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B,
              learning_rate=1e-1,
              lr_step_size=10000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              grad_iter=1):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        
    
        # Init
            
        optimizer = pytorch_optim(model.parameters(),
                learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()
        
        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        print("Start training...")
        for it in range(start_iter, start_iter + train_iter):
            
            input1, mask1, input2, mask2, label = next(self.train_data_loader)

            if torch.cuda.is_available():
                input1 = input1.cuda()
                input2 = input2.cuda()
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()
                label = label.cuda()

            
            logits = model(input1, mask1, input2, mask2)
            
            loss = model.loss(logits, label) / float(grad_iter)
            
            right = model.accuracy(logits, label)
            loss.backward()
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                
                acc = self.eval(model, B, val_iter)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        loss_list = []
        with torch.no_grad():
            for it in range(eval_iter):
                input1, mask1, input2, mask2, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    input1 = input1.cuda()
                    input2 = input2.cuda()
                    mask1 = mask1.cuda()
                    mask2 = mask2.cuda()
                    label = label.cuda()

                logits = model(input1, mask1, input2, mask2)
                loss = model.loss(logits, label)
                loss_list.append(loss.data)
                right = model.accuracy(logits, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | loss: {1:6.5f} | accuracy: {2:3.2f}%'.format(it + 1, sum(loss_list)/len(loss_list), 100 * iter_right / iter_sample) +'\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
