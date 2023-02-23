from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import glob
from transformers import GPT2TokenizerFast
import h5py
import torch
from tqdm import tqdm

class PileH5Dataset(Dataset):
    def __init__(self, datapath, split):
        self.filenames = glob.glob(datapath+'/*.h5')
        nfiles = len(self.filenames)
        print(f"Found {nfiles} files under {split}.")
        for i in sorted(range(len(self.filenames)), reverse=True):
            if i%5 == 0 and split=='train':
                popfile = self.filenames.pop(i)
            if i%10 != 0 and split == 'validation':
                popfile = self.filenames.pop(i)
            
        nfiles = len(self.filenames)
        nlines = 0
        self.nperfile = 65536 # configured from utils/tokenize_and_format.py
        self.window = 2048
        for fn in tqdm(self.filenames, total=nfiles):
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
        mask = torch.randint(0,2,size=(1,self.window), dtype=torch.int16).squeeze(0).bool()
        masked = torch.from_numpy(tokens).clone()
        masked[mask] = 50258
        label = torch.from_numpy(tokens).long()

        return {
                    'input':masked, 
                    'label':label, 
                    'attention_mask':torch.from_numpy(attn)
                }
        
