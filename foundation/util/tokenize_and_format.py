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

def save_file(id_arr, mask_arr, cnt_last_sample, n_files):
    with F(f"{h5dest}/{rank:02d}_{n_files:04d}.h5", 'w') as f:
        f.create_dataset('input_ids', data=id_arr, dtype=np.uint16 )
        f.attrs['last_sample']=cnt_last_sample
        f.attrs['pad_token']=pad_token
        f.attrs['mask_token']=mask_token
        f.attrs['eos_token']=eos_token
        f.attrs['mask_token']=mask_token
        # f.create_dataset('attention_mask', data=mask_arr, dtype=np.int32 )# Dont need attn mask--can just make from token ids in dataloading
    # f.create_dataset('token_type', type_arr, dtype=np.int32 )

world_size= size
parser = ap()
parser.add_argument('--training_files', type=str, nargs='+')
parser.add_argument('--sequence_length', type=int, default=2049)
parser.add_argument('--file_size', type=int, default=65536)
parser.add_argument('--run_number', type=int, default=0)
parser.add_argument('--data_dest', type=str, default='/lus/eagle/projects/candle_aesp/azton/the_h5_pile_l')
args = parser.parse_args()
window = args.sequence_length # maximum sequence length
nperfile = args.file_size # number of samples per file
run_num= args.run_number #number of restarts--make a new directory to avoid accidental overwrites, etc.
h5dest = f'{args.data_dest}{window}_{run_num}'
if not os.path.exists(h5dest):
    os.makedirs(h5dest, exist_ok=True)
local_train_files = [args.training_files[i] for i in range(rank, len(args.training_files), size)]
# type_array = np.zeros((nperfile, window), dtype=np.int32)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = './pretrained_tokenizer')
tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
tokenizer.add_special_tokens({'unk_token':'<|unk|>'})
tokenizer.add_special_tokens({'eos_token':'<|eos|>'})
pad_token = tokenizer.pad_token_id
mask_token = tokenizer.mask_token_id
unk_token = tokenizer.unk_token_id
eos_token = tokenizer.eos_token_id
# tokenizer.save('./pretrained_tokenizer/gpt2_modified_tokenizer.pttok')

def map_function(example):
    rval = tokenizer(example['text'], 
                    pad_to_multiple_of=window,
                    truncation=False,
                    return_tensors="np",)
    return rval
train_ds = load_dataset('json', 
                        data_files=local_train_files, 
                        split='train',
                        # num_proc=4,
                        streaming=True,
                        ).with_format("np")
# train_ds = train_ds.map(map_function, 
#                     batched=True, 
#                     remove_columns=['text','meta'],
#                     )
if run_num > 0:
    prior_dest = f'{args.data_dest}{window}_{run_num-1}'
    prior_files_srt = sorted(glob.glob(f'{prior_dest}/{rank:02d}*.h5'))
    with F(prior_files_srt[-1], 'r') as f:
        nprior_samples = f.attrs['last_sample'] + 1
    n_files = len(prior_files_srt)
    print(f"Rank {rank:02d} skipping {nprior_samples} samples from {n_files} files: {prior_files_srt[-1]}")
    train_ds = train_ds.skip(nprior_samples)
else:
    nprior_samples = 0
    n_files = 0
train_ds = train_ds.map(map_function, 
                    batched=True, 
                    remove_columns=['text','meta'],
                    )
n_samples = 0 # we want a fixed number of samples per hdf5 file
id_array = np.zeros((nperfile, window), dtype=np.uint16)
mask_array = np.zeros((nperfile, window), dtype=np.bool)


bar1 = tqdm(total=7022886*len(local_train_files), position=rank, desc=f"Rank {rank:02d} sample")
bar1.update(nprior_samples)
for j, data in enumerate(train_ds):
    id_arr = np.array(data['input_ids']).astype(np.uint16)
    n_pad = window - id_arr.size % window
    id_arr = np.pad(id_arr, (0,n_pad), mode='constant', constant_values=(pad_token,pad_token))
    id_arr = id_arr.reshape(-1, window)
    for ii, seg in enumerate(id_arr):
        id_array[n_samples] = seg
        n_samples += 1
        if n_samples == nperfile:
            save_file(id_array, None, j+nprior_samples, n_files)
            n_files += 1
            n_samples = 0
    bar1.update()        
    
 