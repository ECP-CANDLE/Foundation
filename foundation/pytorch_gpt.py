"""
    Absolutely raw pytorch version of the lightning GPT.
    This might be necessary since lightning is doing so many things behind curtains.
    le sigh

"""

import torch, os
from argparse import ArgumentParser as ap

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                BackwardPrefetch,
                                                                StateDictType
)
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
# from torchdistx import deferred_init
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
import torch.distributed as td
from datetime import datetime
import time
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import numpy as np

from foundation.util.arguments import parse_arguments
from foundation.util.dataloading import PileH5Dataset
from foundation.util.profiler import LogAndProfiler
from foundation.models.minGPT import (
        GPT, 
        GPTConfig, 
        Block, 
        MLP, 
        CausalSelfAttention
        )
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXMLP
# POLARIS local rank = PMI_LOCAL_RANK
# POLARIS local size = PMI_LOCAL_SIZE
zero = int(os.environ['PMI_RANK']) == 0


def init_distributed(args):

    torch.cuda.set_device(args['local_rank'])
    td.init_process_group(backend=args['backend'], init_method="env://")
    # print(f"Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count()}, on device = {torch.cuda.current_device()}")
    # print(f"Local rank {args['local_rank']}: {torch.cuda.current_device()}")

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
        train_ds = PileH5Dataset(args['datapath'], 'train', args)
        val_ds = PileH5Dataset(args['datapath'], 'validation', args)
        tdsamp = DistributedSampler(train_ds, 
                                    num_replicas=args['world_size'], 
                                    rank=args['global_rank'],
                                    shuffle=True,
                                    )
        vdsamp = DistributedSampler(val_ds, 
                                    num_replicas=args['world_size'], 
                                    rank=args['global_rank'],
                                    shuffle=False,
                                    )
        traindl = DataLoader(train_ds, 
                                batch_size=args['batch_size'], 
                                num_workers=1, 
                                sampler=tdsamp)
        valdl = DataLoader(val_ds, 
                           batch_size=args['batch_size'], 
                           num_workers=1,
                           sampler=vdsamp)
        args['num_train'] = len(traindl)
    else:
        print(f"Loading datasets without use_hdf5 is depracated. Enable use_hdf5 and give me better data")
        raise NotImplemented
    # elif args['training_files'] is not None:
    #     print('Loading huggingface style.  Not recommended, as its slow and has race conditions...')
    #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir = './pretrained_tokenizer')
    #     tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
    #     tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
    #     def map_function(example):
    #         rval = tokenizer(example['text'], 
    #                         max_length=2048, 
    #                         padding='max_length', 
    #                         truncation=True, 
    #                         return_tensors="pt",)
    #         return rval
    #     print('loading datasets... ')
    #     train_ds = load_dataset('json', 
    #                             data_files=args['training_files'], 
    #                             streaming=True, 
    #                             split='train',
    #                             # num_proc=8
    #                             ).with_format("torch")
    #     val_ds = load_dataset('json', 
    #                           data_files=args['validation_files'], 
    #                           streaming=True, split='train',
    #                         #   num_proc=8
    #                         ).with_format("torch")
    #     print('mapping dataset...')
    #     train_ds = train_ds.map(map_function, 
    #                         batched=True, 
    #                         remove_columns=['text','meta'])
    #     val_ds = val_ds.map(map_function, 
    #                         batched=True, 
    #                         remove_columns=['text','meta'])

    #     traindl = get_loader(train_ds)
    #     valdl = get_loader(val_ds)
    print('Dataloaders initialized...')
    return train_ds, traindl, val_ds, valdl

def get_loader(dataset, batch_size, num_workers=2):
    return DataLoader(dataset, 
                           batch_size=batch_size, 
                           num_workers=1, 
                        #    collate_fn=dataset_collator,
                        #    persistent_workers=True
                        #    timeout=10
                        #    prefetch_factor=2,
                           )


def init_model(args):
    if args['model'] == 'nanogpt':
        config = GPTConfig(
            block_size = args['seq_length'], # configured in tokenizer to match GPT-3
            vocab_size = 50304,
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_embd = args['embed_dim'],
            dropout = args['dropout'],
            bias = False,
            init_device = 'meta' # must be true for ~<50B parameters
        )
        model = GPT(config)
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block,}
        )
        ckpt_fn = lambda submod: isinstance(submod, Block)
        non_reent_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=args['cpu_offload'],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )


    elif args['model'] == 'transformergpt':
        config = GPTNeoXConfig(
            vocab_size = 50304,
            hidden_size = args['embed_dim'],
            intermediate_size = args['embed_dim']*4,
            num_hidden_layers = args['num_layers'],
            num_attention_heads = args['num_heads'],
            hidden_act = 'gelu',
            max_position_embeddings = args['seq_length'],
            use_cache = True,
        )
        model = GPTNeoXForCausalLM(config)
        twrap_policy = wrap_policy
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPTNeoXAttention, GPTNeoXMLP}
        )
        ckpt_fn = lambda submod: isinstance(submod, GPTNeoXAttention) or isinstance(submod, GPTNeoXMLP)
        non_reent_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=args['cpu_offload'],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
    
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    model = FSDP(model,
                # param_init_fn=model._init_weights,
                auto_wrap_policy=twrap_policy,
                mixed_precision=mixed_precision,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD, #FULL_SHARD, GRAD_SHARD_OP
                cpu_offload=CPUOffload(offload_params=args['cpu_offload']),
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # bit faster async comms, bit higher memory
                limit_all_gathers=False,
                # use_orig_params=True,
                forward_prefetch=True,

                )
        # num_wpe = torch.zeros(1, device=torch.cuda.current_device())
        # for n, p in model.named_parameters():
        #     print(f"Rank {args['global_rank']}: {n}; parameters size = {p.size()} on cuda:{p.get_device()}")
        #     if 'wpe' in n:
        #         num_wpe += p.size(0)
        # td.all_reduce(num_wpe, td.ReduceOp.SUM)
        # activation checkpointing might save memory, but its
        # super slow to initialize.  Lets only use it if absolutely necessary.
    if args['activation_checkpointing']:
        # print(f"Rank {args['global_rank']} applying activation checkpointing wrapper")
        # start = time.time()
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reent_wrapper, check_fn=ckpt_fn
        )
        # print(f"Rank {args['global_rank']} finished AC wrapper in {time.time()-start} seconds...")
    if args['compile_model']:
        model = torch.compile(model)


    
    print(f"[{args['global_rank']}] initializing {args['model']} completed.")
    return model

class NativeTrainer():
    def __init__(self, model,
                        optimizer,
                        loss_function, 
                        train_dataloader, 
                        val_dataloader, 
                        num_train_iter,
                        num_val_iter,
                        use_profiler,
                        args,
                        best_val_ckpt = True,
                        best_train_ckpt = False,
                        profile_timing = True,
                        profile_memory = True,
                        intra_epoch_write = True
                        ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_function
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.ntrain = num_train_iter
        self.nval = num_val_iter
        self.best_val_ckpt = best_val_ckpt
        self.profile_timing = profile_timing
        self.profile_memory = profile_memory
        self.intra_epoch_write = intra_epoch_write
        if self.intra_epoch_write and self.ntrain > 1000:
            print(f"Warning: intra_epoch_write={self.intra_epoch_write} with num_train_iter={self.ntrain}!  there will be lots of writing to disk for mostly no reason!!!")
            print(f"Raising 'DontDoThatError'")
        self.logprofiler = LogAndProfiler(f"{args['log_dir']}/{args['run_name']}", args['global_rank']) if use_profiler else None

        if self.logprofiler and zero:
            self.logprofiler.save_hyperparams(args)
            self.logprofiler.save_codebase()

        self.val_record_epochloss = 99999
        if zero:
            self.token_bar = tqdm(total=3e11, desc="Tokens", position=0)
        else:
            self.token_bar = None
        """
            TODOS:
                if this is a restart, we need to iterate the dataloaders to catch up to where we left off
                if this is a restart, it would be nice to iterate to a set value as well--that way if we 
                    hit a problematic point, we can iterate past that data.
                Also update all the other metrics and make sure the logger isn't overwriting old information
                    Eg if the .json exists, load it first, then continue
                seed everything for reproducability.
        """
    def run_val_epoch(self, epoch, sanity=False):
        self.model.eval()
        loss = torch.zeros(2).to(torch.cuda.current_device())
        flag_tens = torch.zeros(1).to(torch.cuda.current_device())
        bar = None
        train = self.model.training
        # loader = get_loader(loader)
        if zero:
            bar = tqdm(total=self.nval if not sanity else 2, desc=f"{'Train' if train else 'Validation'} epoch {epoch:02d}", position=2)
        for i, batch in enumerate(self.val_dataloader):
            # print(f"{args['global_rank']}:: {i}")
            nval = self.nval if not sanity else 2
            if i > nval:
                flag_tens += 1
            # td.all_reduce(flag_tens, td.ReduceOp.SUM)
            if flag_tens > 0:
                break
            with torch.no_grad():

                # if self.profile_memory:
                #     self.logprofiler.log_cuda_memory('pre_val_fwd')
                # if self.profile_timing:
                #     self.logprofiler.start('val_fwd')
                logits = self.model(input_ids=batch['input_ids'], labels=batch['label_ids'])
                # if self.profile_timing:
                #     self.logprofiler.finish('val_fwd')
                # if self.profile_memory:
                #     self.logprofiler.log_cuda_memory('post_val_fwd')
                if self.args['model'] == 'nanogpt':
                    thisloss = self.loss_fn(logits, F.one_hot(batch['label_ids'], num_classes=50304).float().to(torch.cuda.current_device()))
                else:
                    thisloss = logits.loss
                loss[0] += thisloss.detach()
                loss[1] += batch['input_ids'].size(0) # batch size
            if zero:
                    
                bar.update()
                bar.set_postfix_str(f"Loss={loss[0].item()/loss[1].item():0.4f}")
        # td.all_reduce(loss, op=td.ReduceOp.SUM)
        # print(f"Rank {args['global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
        if zero: bar.close()
        # self.logprofiler.save_log()
        return loss[0] / (loss[1]+1.0)

    def run_train_epoch(self, epoch):
        # loader = get_loader(loader)
        self.model.train()
        loss = torch.zeros(2, device=torch.cuda.current_device())
        flag_tens = torch.zeros(1, device=torch.cuda.current_device())
        bar = None
        train = self.model.training
        if zero:
            bar = tqdm(total=self.ntrain, desc=f"{'Train' if train else 'Validation'} epoch {epoch:02d}", position=2)
        # print(f"Starting epoch {epoch} on rank {args['global_rank']}...")
        for i, batch in enumerate(self.train_dataloader):
            if i == 0:
                bsize = torch.zeros(1, device=torch.cuda.current_device())
                bsize += batch['input_ids'].size(0)
                td.all_reduce(bsize, td.ReduceOp.SUM)
                self.logprofiler.log_quantity('global_batch_size', bsize.item())
            if i > self.ntrain: 
                flag_tens += 1
            # td.all_reduce(flag_tens, td.ReduceOp.SUM)
            if flag_tens > 0:
                break
            if i > 0 and self.profile_timing:
                self.logprofiler.finish('train/get_batch')
            if self.profile_timing:
                self.logprofiler.start('train/iteration')
            # self.optimizer.zero_grad()
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            for param in self.model.parameters():
                param.grad = None
            if self.profile_timing:
                self.logprofiler.start('cuda_timing')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_train_fwd')
            if self.profile_timing:
                self.logprofiler.finish('cuda_timing')
            if self.profile_timing:
                self.logprofiler.start('train_fwd')
            logits = self.model(input_ids=batch['input_ids'], labels=batch['label_ids'])
            # logits = self.model(batch['masked_input'], batch['input_ids'], self.logprofiler)
            if self.profile_timing:
                self.logprofiler.finish('train_fwd')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('post_train_fwd')
            bz = batch['input_ids'].size(0)
            if self.args['model'] == 'nanogpt':
                thisloss = self.loss_fn(logits, F.one_hot(batch['label_ids'], num_classes=50304).float().to(torch.cuda.current_device()))
            else:
                thisloss = logits.loss
            loss[0] += thisloss.detach()
            loss[1] += batch['input_ids'].size(0) # batch size
            
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_backward')
            if self.profile_timing:
                self.logprofiler.start('backward')
            thisloss.backward()
            if self.profile_timing:
                self.logprofiler.finish('backward')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('post_backward')

            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_optimizer_step')
            if self.profile_timing:
                self.logprofiler.start('optimizer_step')
            self.optimizer.step()
            if self.profile_timing:
                self.logprofiler.finish('optimizer_step')
            if zero:
                self.token_bar.update(self.args['seq_length']*bz*args['world_size'])
                bar.update()
                bar.set_postfix_str(f"Loss={loss[0].item()/loss[1].item():0.4f}")
            if self.intra_epoch_write:
                self.logprofiler.save_log()
            if self.profile_timing:
                self.logprofiler.start('train/get_batch')
            if self.profile_timing:
                self.logprofiler.finish('train/iteration')
        if zero: bar.close()
        # td.all_reduce(loss, op=td.ReduceOp.SUM)
        if self.logprofiler:
            self.logprofiler.save_log()
        # print(f"Rank {args['global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
        return loss[0] / (loss[1]+1.0)

    def save_val_record_checkpoint(self, epochnum, vloss):
        """
            only save the model state dict, eg, for later inference.  use restart checkpoints 
            to save optimizer states, etc, for continuing training.
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                    self.model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = self.model.state_dict()
        if zero:    
            save_name = f"{args['log_dir']}/checkpoints/LLM_nl{args['num_layers']}_nh{args['num_heads']}_ed{args['embed_dim']}_epoch{epochnum:02d}_vloss{vloss:0.4f}"
            torch.save(cpu_state, save_name)

    def save_restart_checkpoint(self, epochnum):
        """
            special checkpoint where we'll save all relevant items for a restart, including optimizer states.  I hope.
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                    self.model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = self.model.state_dict()
        if zero:    
            save_name = f"{args['log_dir']}/checkpoints/LLM_nl{args['num_layers']}_nh{args['num_heads']}_ed{args['embed_dim']}_epoch{epochnum:02d}_vloss{vloss:0.4f}"
            torch.save({'model_state':cpu_state,
                        'optimizer_state':self.optimizer.state_dict(),
                        'record_loss':self.val_record_epochloss,
                        'epoch_number':epochnum,
                        'hyperparameters':self.args}, 
                        save_name)
            
    def train_model(self):
        if zero:
            epoch_bar = tqdm(total=args['max_epochs'], position=1, desc="Epoch")


        for e in range(args['max_epochs']):
            
            if hasattr(self.train_dataloader.dataset, 'set_epoch'):
                self.train_dataloader.dataset.set_epoch(e)# each rank has different ordering, effectively distributed sampling
                self.val_dataloader.dataset.set_epoch(e)

            if e == 0:
                vloss = self.run_val_epoch(0, sanity=True) # little sanity check at epoch 0
                # print(f"{args['global_rank']} returned from sanity check...")

            td.barrier()
            # print(f"{args['global_rank']} starting training epoch")
            tloss = self.run_train_epoch(e)
            if self.logprofiler:
                logloss = tloss.item()
                self.logprofiler.tboard_log_scalar('train/epoch_loss', logloss, e)
                self.logprofiler.tboard_log_scalar('train/epoch_ppl', np.exp(logloss), e)
            # print(f"{args['global_rank']} returned from training epoch")

            td.barrier()
            # print(f"{args['global_rank']} starting validation epoch")
            vloss = self.run_val_epoch(e)
            if self.logprofiler:
                logloss = vloss.item()
                self.logprofiler.tboard_log_scalar('val/epoch_loss', logloss, e)
                self.logprofiler.tboard_log_scalar('val/epoch_ppl', np.exp(logloss), e)
            # self.exp_decay.step()
            # self.plateau_decay.step(vloss.item())
            # print(f"{args['global_rank']} returned from validation epoch")

            td.barrier()
            if vloss < self.val_record_epochloss and self.best_val_ckpt and args['record_ckpts']:
                self.val_record_epochloss = vloss # vloss is already synced across tasks
                self.save_val_record_checkpoint(e, vloss)
            
            run_time = self.logprofiler.check_runtime()
            if self.args['restart_ckpts']:
                if run_time > self.args['wall_time'] - 0.17:
                    self.save_restart_checkpoint()
            if zero:
                epoch_bar.update()
def run_pytorch_training(args):

    args=setup_environment(args, args['run_location'])
    init_distributed(args)
    train_ds, traindl, val_ds, valdl = get_datasets(args)
    model = init_model(args)
    opt = torch.optim.AdamW(model.parameters(), args['learn_rate'])
    loss_fn = nn.CrossEntropyLoss()

    trainer = NativeTrainer(model, 
                                opt,
                                loss_fn,
                                traindl,
                                valdl,
                                args['num_train_iter'],
                                args['num_val_iter'],
                                True,
                                args,
                                best_val_ckpt=True,
                                profile_timing=True,
                                profile_memory=True,
                                intra_epoch_write=True)
    # trainer.logprofiler.log_quantity('wpe_layer', num_wpe)
    # trainer.logprofiler.log_quantity('num_initial_layers', num_wpe/ (args['seq_length']*args['embed_dim']))

    trainer.train_model()

    if zero:
        print('Test run completed!!')
        print(torch.cuda.memory_summary())
    td.destroy_process_group()

    
if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_arguments()
    if (not hasattr(torch, 'compile')) and args['compile_model']:
        args['compile_model'] == False
    run_pytorch_training(args)