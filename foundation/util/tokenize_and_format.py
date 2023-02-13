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

def save_file(id_arr, mask_arr, n):
    with F(f"{h5dest}/.h5", 'w') as f:
        f.create_dataset('input_ids', data=id_arr, dtype=np.int32 )
        f.create_dataset('attention_mask', data=mask_arr, dtype=np.int32 )
    # f.create_dataset('token_type', type_arr, dtype=np.int32 )


window = 2048 # maximum sequence length
nperfile = 65536
h5dest = '/media/azton/work/sample_pile'
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# tokenizer.pre_tokenizer = Whitespace()
# # 
n_samples = 0 # we want a fixed number of samples per hdf5 file
n_files = 0
id_array = np.zeros((nperfile, window), dtype=np.int32)
mask_array = np.zeros((nperfile, window), dtype=np.int32)
# type_array = np.zeros((nperfile, window), dtype=np.int32)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = './pretrained_tokenizer')
tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
tokenizer.save('./pretrained_tokenizer/gpt2_modified_tokenizer.pttok')
with open('/media/azton/work/sample_pile/19.jsonl', 'r') as f:
    for i,l in tqdm(f, total=7022886):
        text = eval(l)['text']
        tokens = tokenizer(text, max_length=window, padding='max_length', truncation=True, return_tensors='np')
        id_array[n_samples] = tokens['input_ids']
        mask_array[n_samples] = tokens['attention_mask']
        # type_array[n_samples] = np.array(tokens['token_type_ids'])
        n_samples += 1
        if n_samples == nperfile:
            # save the file and reset the count
            save_file(id_array, mask_array, n_files)
            n_files += 1
            n_samples = 0
        


