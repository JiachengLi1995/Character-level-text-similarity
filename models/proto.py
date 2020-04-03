import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=64):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.linear = nn.Sequential(nn.Linear(8*self.hidden_size, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, 1),
                                    nn.Sigmoid()
                                    )

    def __dist__(self, input1, input2):

        distance = torch.sqrt(torch.sum((input1-input2)**2, dim=1))
        tmp1 = torch.sqrt(torch.sum(input1 ** 2, dim=1))
        tmp2 = torch.sqrt(torch.sum(input2 ** 2, dim=1))
        distance = distance / (tmp1 + tmp2)

        return distance

    def CoAttention(self, input1, input2, mask1, mask2):

        att = input1 @ input2.transpose(1, 2)
        att = att + mask1 * mask2.transpose(1, 2) * 100
        input1_ = F.softmax(att, 2) @ input2 * mask1
        input2_ = F.softmax(att.transpose(1,2), 2) @ input1 * mask2
        return input1_, input2_

    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def pool(self, inputs):

        inputs,_ = torch.max(inputs, dim=1)

        return inputs

    def forward(self, input1, mask1, input2, mask2):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        input1, input2 = self.sentence_encoder(input1, input2, mask1, mask2) # (B, T, D), where D is the hidden size
        mask1 = mask1[:, :input1.size(1)].unsqueeze(2).float()
        mask2 = mask2[:, :input2.size(1)].unsqueeze(2).float()
           
        input1_, input2_ = self.CoAttention(input1, input2, mask1, mask2)
        input1_ = self.fuse(input1, input1_, 2)
        input2_ = self.fuse(input2, input2_, 2)
        
        enhance_input1 = self.pool(input1_)
        enhance_input2 = self.pool(input2_)

        #logits = self.__dist__(enhance_input1, enhance_input2)
        logits = self.linear(torch.cat([enhance_input1, enhance_input2], dim=1))
        
        return logits

    
    
    
