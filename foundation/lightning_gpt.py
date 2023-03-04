from argparse import ArgumentParser as ap
from datetime import datetime
import time
from functools import partial

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as ptl
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.utilities.meta import (
#     init_meta_context
# )

import torch.distributed as td
from torch.utils.data import DataLoader
import torch, os
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn

from foundation.util.dataloading import PileH5Dataset
from foundation.util.distributed_strategies import (
    setup_strategy,
    setup_environment
)
from foundation.models.minGPT import (
    GPT, 
    GPTConfig, 
    CausalSelfAttention, 
    MLP, 
    Block
)
from util.arguments import parse_arguments

# from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling
# from datasets import load_dataset

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
                    bias = True,
                    init_device='meta' if args['strategy'] == 'fsdp_native' else None
        )
        print('Initializing model...')
        if not args['use_hdf5']:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', 
                                                    cache_dir = './pretrained_tokenizer')
        else:
            self.tokenizer = None
        # self.model = torch.compile(GPT(config))
        # with torch.nn.utils.meta_init():
        self.model = GPT(config)
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
    
    def reset_parameters(self):
        pass

    def forward(self, batch):
        if self.args['training_files']:
            # print(batch)
            logits = self.model(batch['input_ids'], batch['label_ids'])
        else:
            logits = self.model(batch['input_ids'], batch['label_ids'])
        # print(F.one_hot(batch['input_ids'], num_classes=50304).size(), logits.size())
        loss = self.loss_function(logits, F.one_hot(batch['label_ids'], num_classes=50304).float())
        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self(batch)
        self.t0loss += float(loss)
        # self.log('train_step_loss', loss.item())
        if batch_idx % self.nlog == 0  and self.args['time_file']:#and self.trainer.is_global_zero
            # try:
            # rank = torch.distributed.get_rank()
            # print(f'Greetings from rank {rank} on device cuda{torch.cuda.current_device()}!')
            ms = torch.cuda.memory_stats()
            print(f"Rank {self.args['local_rank']}/{self.args['global_rank']} is using {ms['active_bytes.all.peak']/1024**3} GB with {torch.cuda.utilization()} utilization on cuda:{torch.cuda.current_device()}.")
            # except RuntimeError as e:
            # rank = 0
            self.time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
                print(f"Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count}, {torch.cuda.current_device}")
                # print("Environment: ", os.environ)
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





def train_foundation_model(args):
    # logger = TensorBoardLogger(save_dir=args['log_dir'], name=args['run_name'])
    # args = setup_environment(args, 'polaris')
    if args['use_hdf5']:
        train_ds = PileH5Dataset(args['datapath'], 'train', args)
        val_ds = PileH5Dataset(args['datapath'], 'validation', args)
        traindl = DataLoader(train_ds, batch_size=args['batch_size'], num_workers=1)
        valdl = DataLoader(val_ds, batch_size=args['batch_size'], num_workers=1)
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
    # if args['strategy'] == 'fsdp_native':
    #     mixed_precision=MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.bfloat16,
    #         buffer_dtype=torch.bfloat16
    #     )
    #     twrap_policy = partial(
    #         transformer_auto_wrap_policy,
    #         transformer_layer_cls={Block,}
    #     )
    #     strat_args = DDPFullyShardedNativeStrategy(
    #         cluster_environment=PolarisEnvironment(),
    #         parallel_devices=[torch.device('cuda:%d'%d) for d in [0,1,2,3]]*args['num_nodes'],          
    #         cpu_offload=CPUOffload(offload_params=args['cpu_offload']),
    #         sharding_strategy= ShardingStrategy.FULL_SHARD,
    #         mixed_precision=mixed_precision,
    #         auto_wrap_policy= twrap_policy,
    #         backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    #         forward_prefetch=True,
    #         process_group_backend="nccl",
    #         activation_checkpointing=(Block,) if args['activation_checkpointing'] else None
    #     )
    model = LightningGPT(args)
    strat_args = setup_strategy(args, model.model)
    callbatcks = [EarlyStopping(monitor="val_step_loss", mode="min")]

    trainer = ptl.Trainer(
        # plugins=[PolarisEnvironment()],
        logger=None,
        num_nodes = args['num_nodes'],
        enable_progress_bar=True,
        profiler='simple',#AdvancedProfiler('./profile_log', filename=now),
        strategy=strat_args,
        accelerator='gpu',
        devices=-1,
        max_epochs=2,
        default_root_dir=args['log_dir'],
        precision=16,
        max_steps=5000,
        limit_val_batches=args['num_val_iter'],
        limit_train_batches=args['num_train_iter'],
        
    )
    # if trainer.is_global_zero: print('Beginning training run   ')
    trainer.fit(model, traindl, valdl, ckpt_path=None)
    if td.is_global_zero():
        print('Test run completed!!')
        print(torch.cuda.memory_summary())
if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_arguments()
    args = setup_environment(args, args['run_location']) # for consistency
    print(f"Master at {args['master_addr']} from node {args['node_id']}")
    # mp.set_start_method("spawn")
    train_foundation_model(args)