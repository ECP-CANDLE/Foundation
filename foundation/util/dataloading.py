# from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import glob
# from transformers import GPT2TokenizerFast
import h5py
import torch
from tqdm import tqdm

class PileH5Dataset(Dataset):
    def __init__(self, datapath, split, args):
        self.filenames = sorted(glob.glob(datapath+'/*.h5'))
        nfiles = len(self.filenames)
        print(f"Found {nfiles} files under {split}.")
        for i in sorted(range(len(self.filenames)), reverse=True):
            if i%10 > 8 and split=='train':
                popfile = self.filenames.pop(i)
            if i%10 != 8 and split == 'validation':
                popfile = self.filenames.pop(i)
            if i%10 != 9 and split == 'test':
                popfile = self.filenames.pop(i)
        self.args = args
        nfiles = len(self.filenames)
        nlines = 0
        self.nperfile = 65536 # configured from utils/tokenize_and_format.py
        self.window = args['seq_length']
        self.max_window=2048
        for fn in self.filenames:
            nlines += self.nperfile
        self.line_count = nlines
        self.mask_token = 50258
        print(f"Found {nlines} examples for {split}.")

    def __len__(self):
        return self.line_count


    def __getitem__(self, idx):
        fileIdx = idx // self.nperfile
        dataIdx = idx % self.nperfile
        with h5py.File(self.filenames[fileIdx], 'r') as f:
            tokens = f['input_ids'][dataIdx]
            attn = f['attention_mask'][dataIdx]
        if self.args['task'] == 'mask_gen':
            mask = torch.randint(0,2,size=(1,self.window), dtype=torch.bool).squeeze(0)
            masked = torch.from_numpy(tokens).clone()
            masked[mask] = 50258
            label = torch.from_numpy(tokens).long()
        if self.args['task'] == 'next_token' and self.window < self.max_window:
            masked = torch.from_numpy(tokens[:self.window])
            label = torch.from_numpy(tokens[1:self.window+1])
        
        else: 
            print('Somethings wrong in the dataloader nieghborhood')
            exit()

        return {
                    'input_ids': masked.int(), 
                    'label_ids':label.long(), 
                    'attention_mask':torch.from_numpy(attn).bool()
                }
        

