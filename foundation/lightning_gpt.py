from argparse import ArgumentParser as ap
from datetime import datetime
import time
from functools import partial

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as ptl
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.meta import (
#     init_meta_context
# )
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import colo_set_process_memory_fraction
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed
import torch.distributed as td
from torch.distributed.fsdp.wrap import (
   enable_wrap,
   wrap,
)
from torch.utils.data import DataLoader
import torch, os
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ChainedScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from transformers import (
    GPTNeoXForCausalLM,
    GPT2LMHeadModel
)
from util.arguments import parse_arguments
from models.load_models import get_config
from models.load_models import init_model
from util.lr_scheduler import CosineWarmupScheduler
from util.distributed_strategies import PolarisEnvironment
# from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling
# from datasets import load_dataset

zero = int(os.environ['PMI_RANK'])==0

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
        self.save_hyperparameters(args)
        self.strategy = args['strategy']
        self.modelconfig = get_config(args)

        if zero: print('Initializing model...')
        if not args['use_hdf5']:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', 
                                                    cache_dir = './pretrained_tokenizer')
        else:
            self.tokenizer = None
        # self.model = GPT(config)
        self.logfile = args['time_file']
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        with open(self.logfile, 'w') as f:
            f.write(f"{now}\n")
        if zero: print('Initialization complete...')
        self.time = time.time()
        self.loss_function = nn.CrossEntropyLoss()
        self.v0loss = 0
        self.t0loss = 0
        self.nlog = 101
        # Init on meta for colossal and FSDP; deepspeed dont like that
        if args['model'] == 'gpt2_hf':
            with deepspeed.zero.Init():
                self.model = GPT2LMHeadModel(self.modelconfig)
        elif args['model'] == 'gpt_neox':
            with deepspeed.zero.Init():
                self.model = GPTNeoXForCausalLM(self.modelconfig)
        elif args['model'] == 'nanogpt':
            with deepspeed.zero.Init():
                self.model = GPT(self.modelconfig)
    # def configure_sharded_model(self):
    #     if hasattr(self.modelconfig, 'init_device'):
    #         self.modelconfig.init_device = None
    #     # Now init for real, and well add the deepspeed checkpoints
    #     if self.args['strategy'] == 'fsdp_native':
    #         self.model = init_model(self.args)
    #     else:
    #         if self.args['model'] == 'nanogpt':
    #             self.model = GPT(self.modelconfig)
    #         elif self.args['model'] == 'gpt2_hf':
    #             self.model = GPT2LMHeadModel(self.modelconfig)
    #         elif self.args['model'] == 'gpt_neox':
    #             self.model = GPTNeoXForCausalLM(self.modelconfig)


    def reset_parameters(self):
        pass
    def configure_optimizers(self):
        if self.strategy == 'fsdp_native':
            opt = torch.optim.AdamW(self.trainer.model.parameters(), self.args['learn_rate'])
        elif self.strategy == 'deepspeed' and self.args['cpu_offload']:
            opt = DeepSpeedCPUAdam(self.trainer.model.parameters(), self.args['learn_rate'])
        elif self.strategy == 'deepspeed':
            opt = FusedAdam(self.trainer.model.parameters(), self.args['learn_rate'])
        if self.strategy == 'colossalai':
            opt = HybridAdam(self.trainer.model.parameters(), self.args['learn_rate'])
        self.annealer = CosineWarmupScheduler(opt, 
                                    warmup=4000,
                                    max_iters=7e6 * 30.0 / (self.args['batch_size']*td.get_world_size()))
        plateau = ReduceLROnPlateau(opt, 
                                    mode='min', 
                                    factor=0.5, 
                                    patience=2, 
                                    verbose=True, 
                                    threshold=0.0001, 
                                    threshold_mode='rel', 
                                    cooldown=1, 
                                    min_lr=1e-7, 
                                    eps=1e-08,
                                    )
        # self.scheduler = ChainedScheduler([annealer, plateau])

        return {'optimizer': opt, 'lr_scheduler': plateau, 'monitor': 'val/loss'}
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if not self.args['strategy'] == 'colossalai':
            self.annealer.step()
        for i, p in enumerate(self.optimizers().param_groups):
            self.log(f'learn_rate_{i}', p['lr'], on_step=True)
           
    def forward(self, batch):
        if self.args['training_files']:
            # print(batch)
            logits = self.model(batch['input_ids'], batch['label_ids'])
        else:
            logits = self.model(input_ids=batch['input_ids'], labels=batch['label_ids'])
        # print(F.one_hot(batch['input_ids'], num_classes=50304).size(), logits.size())
        if hasattr(logits, 'loss'):
            loss = logits.loss
        else:
            loss = logits[1] # nanoGPT returns (logits, loss)
        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self(batch)
        self.t0loss += float(loss)
        self.log('train/loss', loss.item(), prog_bar=True, logger=True)
        if batch_idx % self.nlog == 0  and self.args['time_file'] and self.current_epoch == 0:#and self.trainer.is_global_zero
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
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % self.nlog == 0 and self.args['time_file']:#and self.trainer.is_global_zero 
            rank = torch.distributed.get_rank()
            # rank=0
            util = '%0.1f'%(torch.cuda.memory_reserved()/1024/1024/1024)
            with open(self.logfile, 'a') as f:
                f.write(f"valid        rank {rank} {batch_idx} {self.v0loss/self.nlog:0.6f} {time.time()-self.time:0.2f} s util: {util} GB\n")
            self.time=time.time()
            self.v0loss = 0
        return loss
    


def train_foundation_model(args):
    logger = TensorBoardLogger(save_dir=args['log_dir'], name=args['run_name'])
    # args = setup_environment(args, 'polaris')
    if args['use_hdf5']:
        train_ds = PileH5Dataset(args['datapath'], 'train', args)
        val_ds = PileH5Dataset(args['datapath'], 'validation', args)
        traindl = DataLoader(train_ds, batch_size=args['batch_size'], num_workers=1)
        valdl = DataLoader(val_ds, batch_size=args['batch_size'], num_workers=1, shuffle=True)
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
    model = LightningGPT(args)
    strat_args = setup_strategy(args)
    callbacks = [EarlyStopping(monitor="val/loss", mode="min"),
                  ModelCheckpoint(monitor="val/loss", 
                                  filename='{epoch}-{val/loss:.2f}',
                                  mode="min", 
                                  save_top_k=3, 
                                  save_last=True)]
    trainer = ptl.Trainer(
        # plugins=[PolarisEnvironment()] if strat_str != 'fsdp_native' else None,
        logger=logger,
        num_nodes = args['num_nodes'],
        enable_progress_bar=True,
        callbacks=callbacks,
        profiler=None,#'simple',#AdvancedProfiler('./profile_log', filename=now),
        strategy=strat_args,
        accelerator='gpu',
        num_sanity_val_steps=0,
        devices=-1,
        max_epochs=args['max_epochs'],
        default_root_dir=args['log_dir'],
        precision=16,
        max_steps=5000,
        val_check_interval=args['num_train_iter'],
        limit_val_batches=args['num_val_iter'],
        limit_train_batches=args['num_train_iter'],
        log_every_n_steps=1,
        
    )
    # if trainer.is_global_zero: print('Beginning training run   ')
    trainer.fit(model, traindl, valdl, ckpt_path=None)
    if zero:
        print('Test run completed!!')
        # print(torch.cuda.memory_summary())
if __name__=='__main__':
    print(f"Executing on pytorch version {torch.__version__}.")
    args = parse_arguments()
    args = setup_environment(args, args['run_location']) # for consistency
    if args['strategy'] == 'deepspeed':
        deepspeed.init_distributed(dist_backend='nccl',
                                   init_method="env://",
                                   rank=args['global_rank'],
                                   world_size=args['world_size'],
                                   auto_mpi_discovery=False,
                                   dist_init_required=True)
    print(f"Master at {args['master_addr']} from node {args['node_id']}")
    # mp.set_start_method("spawn")
    train_foundation_model(args)