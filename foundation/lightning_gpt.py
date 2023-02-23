import pytorch_lightning as ptl
# from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch, os
import torch.multiprocessing as mp
from pytorch_lightning.profilers import AdvancedProfiler
from argparse import ArgumentParser as ap
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   wrap
)
import torch.distributed as td
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy

from foundation.util.dataloading import PileH5Dataset
from foundation.models.minGPT import GPT, GPTConfig, CausalSelfAttention, MLP
from datetime import datetime
import time
from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn

def dataset_collator(data):
    nb = len(data)
    inputs = torch.stack([d['input_ids'] for d in data])
    ni = inputs.size(1)
    maskings = torch.randint(0,2,size=(nb, ni))
    masked_input = inputs.clone()
    try:
        masked_input[maskings] = 50258
    except:
        pass# For sharded models, this might be emtpy.
        #print("Error generating masked inputs: ", maskings.size(), masked_input.size())
    attns = torch.stack([d['attention_mask'] for d in data])
    return {
        'input_ids': inputs,
        'masked_input':masked_input,
        'attention_mask':attns
    }
class LightningGPT(ptl.LightningModule):
    def __init__(self, args):
        # args is a dict, not just the namespace thing
        super().__init__()
        self.args = args
        config = GPTConfig(
                    block_size = 2048, # configured in tokenizer to match GPT-3
                    vocab_size = 50304,
                    n_layer = args['num_layers'],
                    n_head = args['num_heads'],
                    n_embd = args['embed_dim'],
                    dropout = args['dropout'],
                    bias = True
        )
        print('Initializing model...')
        if args['training_files']:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', 
                                                    cache_dir = './pretrained_tokenizer')
        else:
            self.tokenizer = None
        # self.model = torch.compile(GPT(config))
        self.model = wrap(GPT(config))
        self.logfile = args['time_file']
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        with open(self.logfile, 'w') as f:
            f.write(f"{now}\n")
        print('Initialization complete...')
        self.time = time.time()
        self.loss_function = nn.CrossEntropyLoss()
        self.v0loss = 0
        self.t0loss = 0
        self.nlog = 101

    def forward(self, batch):
        if self.args['training_files']:
            # print(batch)
            logits = self.model(batch['masked_input'], batch['input_ids'])
        else:
            logits = self.model(batch['input'], batch['label'])
        # print(F.one_hot(batch['input_ids'], num_classes=50304).size(), logits.size())
        loss = self.loss_function(logits, F.one_hot(batch['input_ids'], num_classes=50304).float())
        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self(batch)
        self.t0loss += float(loss)
        # self.log('train_step_loss', loss.item())
        if batch_idx % self.nlog == 0  and self.args['time_file']:#and self.trainer.is_global_zero
            # try:
            rank = torch.distributed.get_rank()
            # except RuntimeError as e:
            # rank = 0
            util = '%0.1f'%(torch.cuda.memory_reserved()/1024/1024/1024)
            with open(self.logfile, 'a') as f:
                f.write(f"train        rank {rank} {batch_idx} {self.t0loss/self.nlog:0.6f} {time.time()-self.time:0.2f} s util: {util} GB\n")
            self.t0loss = 0
            self.time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
                print(f"Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count}, {torch.cuda.current_device}")
                print("Environment: ", os.environ)
        loss = self(batch)
        self.v0loss += float(loss)
        # self.log('val_ste
        # p_loss', loss.item())
        if batch_idx % self.nlog == 0 and self.args['time_file']:#and self.trainer.is_global_zero 
            rank = torch.distributed.get_rank()
            # rank=0
            util = '%0.1f'%(torch.cuda.memory_reserved()/1024/1024/1024)
            with open(self.logfile, 'a') as f:
                f.write(f"valid        rank {rank} {batch_idx} {self.t0loss/self.nlog:0.6f} {time.time()-self.time:0.2f} s util: {util} GB\n")
            self.time=time.time()
            self.v0loss = 0
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.trainer.model.parameters(), self.args['learn_rate'])
def setup_environment(args, machine):
    if machine == 'polaris':
        os.environ['RANK'] = os.environ['PMI_RANK']# global 
        os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
        args['world_size'] = int(os.environ['PMI_SIZE'])
        args['global_rank'] = int(os.environ['PMI_RANK'])
        args['local_rank'] = int(os.environ['PMI_LOCAL_RANK']) # wraparound since LOCAL_RANK is actually global?? WRT/ 
        args['local_size'] = int(os.environ['PMI_LOCAL_SIZE'])
        args['backend'] = 'nccl'
        args['num_nodes'] = args['world_size'] // args['local_size']
    else:
        print(f"{machine} not recognized, cannot initialize distributed environment")
        exit()
    return args
def parse_args():
    par = ap()

    par.add_argument('--num-gpus', '-gpus', type=int, default=None)

    par.add_argument('--model', type=str, 
                        default=None, 
                        help="Model to process", 
                        choices=['nanogpt'])
    par.add_argument('--run_location', '-location', 
                                type=str, 
                                default=None, 
                                help="where are we running? at home or on a remote (sc)? Will skip .distributed stuff at home", 
                                choices=['home','sc'])
    par.add_argument('--max_epochs', type=int, default=5)
    par.add_argument('--log_dir', type=str, default='./')
    par.add_argument('--time_file', type=str, default=None)
    par.add_argument('--run_name', type=str, default=None)
    par.add_argument('--learn_rate', type=float, default=None)
    
    par.add_argument('--num_layers','-nl', type=int, default=12)
    par.add_argument('--num_heads', '-nh', type=int, default=12)
    par.add_argument('--embed_dim', '-ed', type=int, default=768)
    par.add_argument('--dropout', type=float, default=0.0)
    par.add_argument('--use_hdf5', type=str, default='False')
    par.add_argument('--datapath', type=str, default=None)
    par.add_argument('--training_files', type=str, nargs='*', default=None)
    par.add_argument('--validation_files', type=str, nargs='*', default=None)
    par.add_argument('--local_rank', type=int, default=None)
    par=par.parse_args()
    par.use_hdf5 = True if par.use_hdf5.lower()=='true' else False
    args = vars(par)
    for k, v in args.items():
        print(f"{k}:{v}")

    args['strategy'] = 'fsdp_native'
    return args




def train_foundation_model(args):
    # logger = TensorBoardLogger(save_dir=args['log_dir'], name=args['run_name'])
    args = setup_environment(args, 'polaris')
    if args['use_hdf5']:
        train_ds = PileH5Dataset(args['datapath'], 'train')
        val_ds = PileH5Dataset(args['datapath'], 'validation')
        traindl = DataLoader(train_ds, batch_size=1, num_workers=0)
        valdl = DataLoader(val_ds, batch_size=1, num_workers=0)
        args['num_train'] = len(traindl)
    elif args['training_files'] is not None:
        print('loading tokenizer, etc...')
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = './pretrained_tokenizer')
        tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
        tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
        def map_function(example):
            rval = tokenizer(example['text'], 
                            max_length=2048, 
                            padding='max_length', 
                            truncation=True, 
                            return_tensors="pt",)
            return rval
        print('loading datasets... ')
        train_ds = load_dataset('json', data_files=args['training_files'], streaming=True, split='train').with_format("torch")
        val_ds = load_dataset('json', data_files=args['validation_files'], streaming=True, split='train').with_format("torch")
        print('mapping dataset...')
        train_ds = train_ds.map(map_function, 
                            batched=True, 
                            remove_columns=['text','meta'])
        val_ds = val_ds.map(map_function, 
                            batched=True, 
                            remove_columns=['text','meta'])

        traindl = DataLoader(train_ds, batch_size=1, num_workers=0, collate_fn=dataset_collator)
        valdl = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=dataset_collator)
    print('Dataloaders initialized...')
    if args['strategy'] == 'fsdp_native':
        strat_args = DDPFullyShardedNativeStrategy(
            cpu_offload=CPUOffload(offload_params=False),
            # sharding_strategy= FULL_SHARD,
            auto_wrap_policy= wrap_policy,
            # backward_prefetch=None,
            # forward_prefetch=None,
            process_group_backend="nccl",
            # activation_checkpointing=MLP
        )
    model = LightningGPT(args)
    callbatcks = [EarlyStopping(monitor="val_step_loss", mode="min")]
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    nnodes = os.environ['NNODES']

    trainer = ptl.Trainer(
        logger=None,
        enable_progress_bar=True,
        profiler='simple',#AdvancedProfiler('./profile_log', filename=now),
        strategy=strat_args,
        accelerator='gpu',
        devices=-1,
        num_nodes=nnodes,
        max_epochs=2,
        default_root_dir=args['log_dir'],
        precision=16,
        max_steps=5000,
        limit_val_batches=1000,
        limit_train_batches=5000
    )

    # if trainer.is_global_zero: print('Beginning training run   ')
    trainer.fit(model, traindl, valdl, ckpt_path=None)

if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_args()
    # mp.set_start_method("spawn")
    train_foundation_model(args)