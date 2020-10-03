import os
import json

class Vocab():
    """
    FewRel Dataset
    """
    def __init__(self, name, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
    
    def prepare_vocab(self):

        v_set = set()

        abb_list = list(self.json_data.keys())
        
        for abb in abb_list:
            v_set |= set(list(abb.lower()))
            full_text_list = self.json_data[abb]
            for text in full_text_list:
                v_set |= set(list(text.lower()))
        
        v_list = sorted(list(v_set))
        id2char = ['[PAD]', '[UNK]']
        id2char = id2char + v_list

        char2id = dict()

        for i, char in enumerate(id2char):
            char2id[char] = i
        
        return char2id

    
