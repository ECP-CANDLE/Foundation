"""
    Absolutely raw pytorch version of the lightning GPT.
    This might be necessary since lightning is doing so many things behind curtains.
    le sigh

"""

from torch.utils.data import DataLoader
import torch, os
import torch.multiprocessing as mp
from argparse import ArgumentParser as ap
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                StateDictType
)
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
import torch.distributed as td

from foundation.util.dataloading import PileH5Dataset
from foundation.models.minGPT import GPT, GPTConfig, Block
from datetime import datetime
import time
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from lightning_gpt import dataset_collator, parse_args
from transformers import GPT2TokenizerFast
from functools import partial
import numpy as np
# POLARIS local rank = PMI_LOCAL_RANK
# POLARIS local size = PMI_LOCAL_SIZE
zero = int(os.environ['PMI_RANK']) == 0
def init_distributed(args):

    torch.cuda.set_device(args['local_rank'])
    td.init_process_group(backend=args['backend'], init_method="env://")
    print(f"Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count()}, on device = {torch.cuda.current_device()}")
    print(f"Local rank {args['local_rank']}: {torch.cuda.current_device()}")

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
def get_datasets(args):
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
        train_ds = load_dataset('json', 
                                data_files=args['training_files'], 
                                streaming=True, 
                                split='train',
                                # num_proc=8
                                ).with_format("torch")
        val_ds = load_dataset('json', 
                              data_files=args['validation_files'], 
                              streaming=True, split='train',
                            #   num_proc=8
                            ).with_format("torch")
        print('mapping dataset...')
        train_ds = train_ds.map(map_function, 
                            batched=True, 
                            remove_columns=['text','meta'])
        val_ds = val_ds.map(map_function, 
                            batched=True, 
                            remove_columns=['text','meta'])

        traindl = get_loader(train_ds)
        valdl = get_loader(val_ds)
    print('Dataloaders initialized...')
    return train_ds, traindl, val_ds, valdl

def get_loader(dataset):
    return DataLoader(dataset, 
                           batch_size=2, 
                           num_workers=2, 
                           collate_fn=dataset_collator,
                        #    persistent_workers=True
                        #    timeout=10
                        #    prefetch_factor=2,
                           )
def init_model(args):
    if args['model'] == 'nanogpt':
        config = GPTConfig(
            block_size = 2048, # configured in tokenizer to match GPT-3
            vocab_size = 50304,
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_embd = args['embed_dim'],
            dropout = args['dropout'],
            bias = True
        )
        model = GPT(config)
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block,}
        )
        ckpt_fn = lambda submod: isinstance(submod, Block)
        non_reent_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        model = FSDP(model,
                    auto_wrap_policy=twrap_policy,
                    mixed_precision=mixed_precision,
                    device_id=torch.cuda.current_device(),
                    sharding_strategy=ShardingStrategy.FULL_SHARD, #FULL_SHARD, GRAD_SHARD_OP
                    cpu_offload=CPUOffload(offload_params=False)
                    )
        # activation checkpointing might save memory, but its
        # super slow to initialize.  Lets only use it if absolutely necessary.

        # apply_activation_checkpointing(
        #     model, checkpoint_wrapper_fn=non_reent_wrapper, check_fn=ckpt_fn
        # )

    else:
        print(f"Model {args['model']} is not implemented")
        exit()
    print(f"[{args['global_rank']}] initializing {args['model']} completed.")
    return model

def run_val_epoch(model, loader, num_iter_max, loss_function, epoch):
    model.eval()
    loss = torch.zeros(2).to(torch.cuda.current_device())
    flag_tens = torch.zeros(1).to(torch.cuda.current_device())
    bar = None
    train = model.training
    # loader = get_loader(loader)
    if zero:
        bar = tqdm(total=num_iter_max, desc=f"{'Train' if train else 'Validation'} epoch {epoch:02d}")
    for i, batch in enumerate(loader):
        # print(f"{args['global_rank']}:: {i}")
        if i > num_iter_max: # counting batches since... whatever.
            flag_tens += 1
        td.all_reduce(flag_tens, td.ReduceOp.SUM)
        if flag_tens > 0:
            break
        with torch.no_grad():
            logits = model(batch['masked_input'], batch['input_ids'])
            thisloss = loss_function(logits, F.one_hot(batch['input_ids'], num_classes=50304).float().to(torch.cuda.current_device()))
            loss[0] += thisloss.detach()
            loss[1] += batch['masked_input'].size(0) # batch size
        if zero:
            if i%1000 == 1:
                bar.write(f"Val Epoch {epoch:01d} : {i:06d} -- rank {args['global_rank']} using {torch.cuda.memory_allocated()/1024**3}")
                
            bar.update()
            bar.set_postfix_str(f"Loss={loss[0].item()/loss[1].item():0.4f}")
    # td.all_reduce(loss, op=td.ReduceOp.SUM)
    print(f"Rank {args['global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
    if zero: bar.close()
    return loss[0] / (loss[1]+1.0)

def run_train_epoch(model, loader, num_iter_max, optimizer, loss_function, epoch):
    # loader = get_loader(loader)
    model.train()
    loss = torch.zeros(2).to(torch.cuda.current_device())
    flag_tens = torch.zeros(1).to(torch.cuda.current_device())
    bar = None
    train = model.training
    if zero:
        bar = tqdm(total=num_iter_max, desc=f"{'Train' if train else 'Validation'} epoch {epoch:02d}")
    print(f"Starting epoch {epoch} on rank {args['global_rank']}...")
    for i, batch in enumerate(loader):
        # print(f"{args['global_rank']}:: {i}")
        if i > num_iter_max: # counting batches since... whatever.
            flag_tens += 1
        td.all_reduce(flag_tens, td.ReduceOp.SUM)
        if flag_tens > 0:
            break
        if train:
            optimizer.zero_grad()
        logits = model(batch['masked_input'], batch['input_ids'])
        thisloss = loss_function(logits, F.one_hot(batch['input_ids'], num_classes=50304).float().to(torch.cuda.current_device()))
        loss[0] += thisloss.detach()
        loss[1] += batch['masked_input'].size(0) # batch size
        if train:
            thisloss.backward()
            optimizer.step()
        if zero:
            if i%1000 == 1:
                bar.write(f"Train Epoch {epoch:01d} : {i:06d} -- rank {args['global_rank']} using {torch.cuda.memory_allocated()/1024**3}")
                
            bar.update()
            bar.set_postfix_str(f"Loss={loss[0].item()/loss[1].item():0.4f}")
    if zero: bar.close()
    # td.all_reduce(loss, op=td.ReduceOp.SUM)
    print(f"Rank {args['global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
    return loss[0] / (loss[1]+1.0)

def save_val_record_checkpoint(model, epochnum, vloss):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
    if zero:    
        save_name = f"checkpoints/LLM_nl{args['num_layers']}_nh{args['num_heads']}_ed{args['embed_dim']}_epoch{epochnum:02d}_vloss{vloss:0.4f}"
        torch.save(cpu_state, save_name)

def run_pytorch_training(args):

    # for k, v in os.environ.items():
    #     if 'PMI' in k or 'PBS' in k:
    #         print(f"{k}: {v}")
    args=setup_environment(args, 'polaris')
    init_distributed(args)
    train_ds, traindl, val_ds, valdl = get_datasets(args)
    model = init_model(args)
    print(f"Rank {args['global_rank']} using {torch.cuda.memory_allocated()/1024**3}")
    for name, params in model.named_parameters():
        print(f"Rank {args['global_rank']} ::: {name} ::: device {params}")
    # print(model)
    opt = torch.optim.AdamW(model.parameters(), args['learn_rate'])
    loss_fn = nn.CrossEntropyLoss()
    tlosses = []
    vlosses = []

    for e in range(args['max_epochs']):
        train_ds.set_epoch(e)# each rank has different ordering, effectively distributed sampling
        val_ds.set_epoch(e)

        if e == 0:
            vloss = run_val_epoch(model, valdl, 10, loss_fn,e) # little sanity check at epoch 0
            print(f"{args['global_rank']} returned from sanity check...")

        td.barrier()
        print(f"{args['global_rank']} starting training epoch")
        tloss = run_train_epoch(model, traindl, 150, opt, loss_fn,e)
        tlosses.append(tloss.item())
        print(f"{args['global_rank']} returned from training epoch")

        td.barrier()
        print(f"{args['global_rank']} starting validation epoch")
        vloss = run_val_epoch(model, valdl, 50, loss_fn,e)
        vlosses.append(vloss)
        print(f"{args['global_rank']} returned from validation epoch")

        td.barrier()
        
        save_val_record_checkpoint(model, e, vloss)
    if zero:
        print('Test run completed!!')
    td.destroy_process_group()
if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_args()
    run_pytorch_training(args)