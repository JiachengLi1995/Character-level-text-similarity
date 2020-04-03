import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.abbreviation = list(self.json_data.keys())
        self.full_text = []
        for val in list(self.json_data.values()):
            self.full_text+=val
        self.N = N
        self.encoder = encoder

    def __getraw__(self, abbs, target):
        input1, mask1, input2, mask2  = self.encoder.tokenize(abbs, target)
        return input1, mask1, input2, mask2 

    def __getitem__(self, index):
        abbs = list(random.sample(self.abbreviation, self.N))
        target_text = [] + abbs
        for abb in abbs:
            target_text+=self.json_data[abb]
        label = []
        
        if random.random()>0.5:
            target = random.sample(target_text, 1)[0]
            label.append(1)
        else:
            target = random.sample(self.full_text, 1)[0]
            while target in target_text:
                target = random.sample(self.full_text, 1)[0]
            label.append(0)

        input1, mask1, input2, mask2 = self.__getraw__(abbs, target)

        input1 = torch.tensor(input1).long()
        mask1 = torch.tensor(mask1).long()
        input2 = torch.tensor(input2).long()
        mask2 = torch.tensor(mask2).long()

        return input1, mask1, input2, mask2, label
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    
    input1, mask1, input2, mask2, label = zip(*data)
    batch_input1 = torch.stack(input1, 0)
    batch_mask1 = torch.stack(mask1, 0)
    batch_input2 = torch.stack(input2, 0)
    batch_mask2 = torch.stack(mask2, 0)
    batch_label = torch.tensor(label)
    return batch_input1, batch_mask1, batch_input2, batch_mask2, batch_label

def get_loader(name, encoder, N, batch_size, num_workers=8, collate_fn=collate_fn, root='./data'):
    dataset = FewRelDataset(name, encoder, N, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)
