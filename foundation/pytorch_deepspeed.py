"""
    Absolutely raw pytorch version of the lightning GPT.
    This might be necessary since lightning is doing so many things behind curtains.
    le sigh

    This version uses DeepSpeed!  Now with ND paralellism!


"""

import torch, os
from argparse import ArgumentParser as ap
import deepspeed

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                StateDictType
)

import torch.distributed as td
from datetime import datetime
import time
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import numpy as np

from foundation.models.load_models import init_ds_model
from foundation.util.deepspeed_arguments import parse_arguments
from foundation.util.dataloading import PileH5Dataset
from foundation.util.profiler import LogAndProfiler
from foundation.util.distributed_strategies import setup_environment
from foundation.util.prepare_deepspeed import (
        prepare_optimizer_parameters
)
# POLARIS local rank = PMI_LOCAL_RANK
# POLARIS local size = PMI_LOCAL_SIZE
zero = int(os.environ['PMI_RANK']) == 0


def init_distributed(args):

    deepspeed.init_distributed(dist_backend='nccl',
                                 dist_init_required=True,
                                 rank=args.global_rank,
                                 world_size=args.world_size,
                                 auto_mpi_discovery=False,
    )    
    # print(f"Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count()}, on device = {torch.cuda.current_device()}")
    # print(f"Local rank {args.local_rank']}: {torch.cuda.current_device()}")


def get_datasets(args):
    if args.use_hdf5:
        train_ds = PileH5Dataset(args.datapath, 'train', args)
        val_ds = PileH5Dataset(args.datapath, 'validation', args)
        tdsamp = DistributedSampler(train_ds, 
                                    num_replicas=args.world_size, 
                                    rank=args.global_rank,
                                    shuffle=True,
                                    )
        vdsamp = DistributedSampler(val_ds, 
                                    num_replicas=args.world_size, 
                                    rank=args.global_rank,
                                    shuffle=False,
                                    )
        traindl = DataLoader(train_ds, 
                                batch_size=args.batch_size, 
                                num_workers=1, 
                                sampler=tdsamp)
        valdl = DataLoader(val_ds, 
                           batch_size=args.batch_size, 
                           num_workers=1,
                           sampler=vdsamp)
        args.num_train = len(traindl)
    else:
        print(f"Loading datasets without use_hdf5 is depracated. Enable use_hdf5 and give me better data")
        raise NotImplemented

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




class NativeTrainer():
    def __init__(self, model,
                        optimizer,
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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.ntrain = num_train_iter
        self.nval = num_val_iter
        self.best_val_ckpt = best_val_ckpt
        self.profile_timing = profile_timing
        self.profile_memory = profile_memory
        self.intra_epoch_write = intra_epoch_write
        if self.intra_epoch_write and self.ntrain > 1000:
            print(f"Warning: intra_epoch_write={self.intra_epoch_write} with num_train_iter={self.ntrain}!  there will be lots of writing to disk for mostly no reason from all ranks!!!")
            print(f"Raising 'DontDoThatError' (But its fake, we'll let your bad decision slide.  Your sys admin might not be so forgiving.)")
        self.logprofiler = LogAndProfiler(f"{args.log_dir}/{args.run_name}", args.global_rank) if use_profiler else None

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

    def forward(self, batch):
        logits = self.model(input_ids=batch['input_ids'].to(torch.cuda.current_device()), labels=batch['label_ids'].to(torch.cuda.current_device()))
        return logits

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
            # print(f"{args.global_rank}:: {i}")
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
                logits = self.forward(batch)
                # if self.profile_timing:
                #     self.logprofiler.finish('val_fwd')
                # if self.profile_memory:
                #     self.logprofiler.log_cuda_memory('post_val_fwd')
                if type(logits)==tuple:
                    thisloss = logits[1]
                else:
                    thisloss = logits.loss
                loss[0] += thisloss.detach()
                loss[1] += batch['input_ids'].size(0) # batch size
            if zero:
                    
                bar.update()
                bar.set_postfix_str(f"Loss={loss[0].item()/(1+loss[1].item()):0.4f}")
        # td.all_reduce(loss, op=td.ReduceOp.SUM)
        # print(f"Rank {args.global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
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
        # print(f"Starting epoch {epoch} on rank {args.global_rank']}...")
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
            if self.profile_timing:
                self.logprofiler.start('cuda_timing')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_train_fwd')
            if self.profile_timing:
                self.logprofiler.finish('cuda_timing')
            if self.profile_timing:
                self.logprofiler.start('train_fwd')
            logits = self.forward(batch)
            # logits = self.model(batch['masked_input'], batch['input_ids'], self.logprofiler)
            if self.profile_timing:
                self.logprofiler.finish('train_fwd')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('post_train_fwd')
            bz = batch['input_ids'].size(0)
            if type(logits)==tuple:
                thisloss = logits[1]
            else:
                thisloss = logits.loss
            loss[0] += thisloss.detach()
            loss[1] += batch['input_ids'].size(0) # batch size
            
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_backward')
            if self.profile_timing:
                self.logprofiler.start('backward')
            self.model.backward(thisloss)
            if self.profile_timing:
                self.logprofiler.finish('backward')
            if self.profile_memory:
                self.logprofiler.log_cuda_memory('post_backward')

            if self.profile_memory:
                self.logprofiler.log_cuda_memory('pre_optimizer_step')
            if self.profile_timing:
                self.logprofiler.start('optimizer_step')
            self.model.step()
            if self.profile_timing:
                self.logprofiler.finish('optimizer_step')
            if zero:
                self.token_bar.update(self.args.seq_length*bz*args.world_size)
                bar.update()
                bar.set_postfix_str(f"Loss={loss[0].item()/(loss[1].item()+1):0.4f}")
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
        # print(f"Rank {args.global_rank']} finished epoch {epoch} with {loss[0]/(loss[1]+1.)}...")
        return loss[0] / (loss[1]+1.0)

    def save_val_record_checkpoint(self, epochnum, vloss):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {'epoch': epoch,
                                'last_global_step': last_global_step,
                                'last_global_data_samples': last_global_data_samples}
        # Add extra kwargs too
        checkpoint_state_dict.update(kwargs)

        success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)

        return

    def save_restart_checkpoint(self, PATH, ckpt_id, epochnum, last_global_step):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {'epoch': epoch,
                                'last_global_step': last_global_step,
                                'last_global_data_samples': last_global_data_samples}
        # Add extra kwargs too
        checkpoint_state_dict.update(kwargs)

        success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)

        return
    def load_training_checkpoint(args, model, PATH, ckpt_id):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """

        _, checkpoint_state_dict = model.load_checkpoint(PATH, ckpt_id)

        epoch = checkpoint_state_dict['epoch']
        last_global_step = checkpoint_state_dict['last_global_step']
        last_global_data_samples = checkpoint_state_dict['last_global_data_samples']
        del checkpoint_state_dict
        return (epoch, last_global_step, last_global_data_samples)        
    def train_model(self):
        if zero:
            epoch_bar = tqdm(total=args.max_epochs, position=1, desc="Epoch")


        for e in range(args.max_epochs):
            
            if hasattr(self.train_dataloader.dataset, 'set_epoch'):
                self.train_dataloader.dataset.set_epoch(e)# each rank has different ordering, effectively distributed sampling
                self.val_dataloader.dataset.set_epoch(e)

            if e == 0:
                vloss = self.run_val_epoch(0, sanity=True) # little sanity check at epoch 0
                # print(f"{args.global_rank']} returned from sanity check...")

            td.barrier()
            # print(f"{args.global_rank']} starting training epoch")
            tloss = self.run_train_epoch(e)
            if self.logprofiler:
                logloss = tloss.item()
                self.logprofiler.tboard_log_scalar('train/epoch_loss', logloss, e)
                self.logprofiler.tboard_log_scalar('train/epoch_ppl', np.exp(logloss), e)
            # print(f"{args.global_rank']} returned from training epoch")

            td.barrier()
            # print(f"{args.global_rank']} starting validation epoch")
            vloss = self.run_val_epoch(e)
            if self.logprofiler:
                logloss = vloss.item()
                self.logprofiler.tboard_log_scalar('val/epoch_loss', logloss, e)
                self.logprofiler.tboard_log_scalar('val/epoch_ppl', np.exp(logloss), e)
            # self.exp_decay.step()
            # self.plateau_decay.step(vloss.item())
            # print(f"{args.global_rank']} returned from validation epoch")

            td.barrier()
            if vloss < self.val_record_epochloss and self.best_val_ckpt and args.record_ckpts:
                self.val_record_epochloss = vloss # vloss is already synced across tasks
                self.save_val_record_checkpoint(e, vloss)
            
            run_time = self.logprofiler.check_runtime()
            if self.args.restart_ckpts:
                if run_time > self.args.wall_time - 0.17:
                    self.save_restart_checkpoint()
            if zero:
                epoch_bar.update()
def run_pytorch_training(args):
    args=setup_environment(args, args.run_location)
    init_distributed(args)
    train_ds, traindl, val_ds, valdl = get_datasets(args)

    model = init_ds_model(args)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, 
                                                        model=model, 
                                                        model_parameters=parameters,
                                                        )

    trainer = NativeTrainer(model_engine, 
                                optimizer,
                                traindl,
                                valdl,
                                args.num_train_iter,
                                args.num_val_iter,
                                True,
                                args,
                                best_val_ckpt=True,
                                profile_timing=True,
                                profile_memory=True,
                                intra_epoch_write=True)
    # trainer.logprofiler.log_quantity('wpe_layer', num_wpe)
    # trainer.logprofiler.log_quantity('num_initial_layers', num_wpe/ (args.seq_length*args.embed_dim))

    trainer.train_model()

    if zero:
        print('Test run completed!!')
        print(torch.cuda.memory_summary())
    td.destroy_process_group()

    
if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_arguments(deepspeed_args=True)
    if (not hasattr(torch, 'compile')) and args.compile_model:
        args.compile_model == False
    run_pytorch_training(args)