"""
    The pile is a mess of hard-to-read stuff.  Tokenization is also a time-consuming processing step, 
    so this will perform multiple tasks:
    1) Pull data from the pile, extracting text and possible labels, etc.
    2) tokenize the text
    3) create train/test/val splits
    4) store these things as HDF5 files for easier indexing access.

    Now we dont need tokenizers in the loop!    
"""
from transformers import GPT2TokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
import numpy as np
from h5py import File as F
from tqdm import tqdm
from datasets import load_dataset
from argparse import ArgumentParser as ap
import os, glob, math
rank = int(os.environ['PMI_RANK'])
size = int(os.environ['PMI_SIZE'])

def save_file(id_arr, mask_arr, n):
    with F(f"{h5dest}/{rank:01d}_{n:04d}.h5", 'w') as f:
        f.create_dataset('input_ids', data=id_arr, dtype=np.int32 )
        f.create_dataset('attention_mask', data=mask_arr, dtype=np.int32 )
    # f.create_dataset('token_type', type_arr, dtype=np.int32 )

mask_token=50258
world_size= 64
parser = ap()
parser.add_argument('--training_files', type=str, nargs='+')
args = parser.parse_args()
window = 2048 # maximum sequence length
nperfile = 65536
run_num=1 #number of restarts--make a new directory to avoid accidental overwrites, etc.
h5dest = f'/lus/eagle/projects/candle_aesp/azton/the_h5_pile_{run_num}'
if not os.path.exists(h5dest):
    os.makedirs(h5dest, exist_ok=True)
local_train_files = [args.training_files[i] for i in range(rank, len(args.training_files), size)]
# type_array = np.zeros((nperfile, window), dtype=np.int32)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = './pretrained_tokenizer')
tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
# tokenizer.save('./pretrained_tokenizer/gpt2_modified_tokenizer.pttok')

def map_function(example):
    rval = tokenizer(example['text'], 
                    pad_to_multiple_of=2048,
                    truncation=False,
                    return_tensors="np",)
    return rval
train_ds = load_dataset('json', 
                        data_files=local_train_files, 
                        streaming=True, 
                        split='train',
                        # num_proc=8
                        ).with_format("np")

train_ds = train_ds.map(map_function, 
                    batched=True, 
                    remove_columns=['text','meta'],
                    )
prior_dest = f'/lus/eagle/projects/candle_aesp/azton/the_h5_pile'
prior_files = glob.glob(f"{h5dest[:-2]}/{rank}_*.h5") # previously only written by this rank
nprior_samples = len(prior_files)*nperfile

n_samples = 0 # we want a fixed number of samples per hdf5 file
n_files = 0
id_array = np.zeros((nperfile, window), dtype=np.uint16)
mask_array = np.zeros((nperfile, window), dtype=np.bool)


bar0 = tqdm(total=nperfile, position=rank+size, desc=f'Rank {rank:02d} file progress')
bar1 = tqdm(total=7022886*len(local_train_files), position=rank, desc=f"Rank {rank:02d} sample")
for j, data in enumerate(train_ds):
    this_sample=n_samples+n_files*nperfile
    log2write = this_sample > nprior_samples

    if log2write:
        id_arr = np.array(data['input_ids']).astype(np.uint16)
        n_pad = window - id_arr.size % window
        id_arr = np.pad(id_arr, (0,n_pad), mode='constant', constant_values=(mask_token,mask_token))
        mask_arr = np.array(data['attention_mask']).astype(np.bool)
        mask_arr = np.pad(mask_arr, (0,n_pad), mode='constant', constant_values=(0,0))
        id_arr = id_arr.reshape(-1, window)
        mask_arr = mask_arr.reshape(-1, window)        
        for ii, seg in enumerate(id_arr):
            id_array[n_samples] = seg
            mask_array[n_samples] = mask_arr[ii]
            n_samples += 1
            bar0.update()
            if n_samples == nperfile:
                save_file(id_array, mask_array, n_files)
                n_files += 1
                bar0.update(-n_samples)
                n_samples = 0
    else:
        nlocal = math.ceil(len(data['input_ids'])//2048.)
        n_samples += nlocal
        bar0.update(nlocal)
        if n_samples >= nperfile:
            bar0.update(-n_samples)
            n_samples = 0
            n_files += 1
    bar1.update()        
    
# the last little bit gets skipped =/


# with open('/media/azton/work/sample_pile/19.jsonl', 'r') as f:
#     for i,l in tqdm(f, total=7022886):
#         text = eval(l)['text']
#         tokens = tokenizer(text, max_length=window, padding='max_length', truncation=True, return_tensors='np')
#         id_array[n_samples] = tokens['input_ids']
#         mask_array[n_samples] = tokens['attention_mask']
#         # type_array[n_samples] = np.array(tokens['token_type_ids'])
#         n_samples += 1
#         if n_samples == nperfile:
#             # save the file and reset the count
#             save_file(id_array, mask_array, n_files)
#             n_files += 1
#             n_samples = 0
        


