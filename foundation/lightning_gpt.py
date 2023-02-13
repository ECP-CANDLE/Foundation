import pytorch_lightning as ptl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser as ap
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy

from foundation.util.dataloading import PileH5Dataset
from foundation.models.minGPT import GPT, GPTConfig


class LightningGPT(ptl.LightningModule):
    def __init__(self, args):
        # args is a dict, not just the namespace thing
        super(LightningGPT, self).__init__()
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

        self.model = torch.compile(GPT(config))


    def forward(self, batch):
        logits, loss = self.model(batch['input'], batch['label'])
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_step_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_step_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.trainer.model.parameters(), self.args['learn_rate'])

def parse_args():
    par = ap()

    par.add_argument('--model', type=str, 
                        default=None, 
                        help="Model to process", 
                        choices=['nanogpt'])
    par.add_argument('--log_dir', type=str, default='./')
    par.add_argument('--run_name', type=str, default=None)
    par.add_argument('--learn_rate', type=float, default=None)
    
    par.add_argument('--num_layers','-nl', type=int, default=12)
    par.add_argument('--num_heads', '-nh', type=int, default=12)
    par.add_argument('--embed_dim', '-ed', type=int, default=768)
    par.add_argument('--dropout', type=float, default=0.0)
    
    par.add_argument('--datapath', type=str, default=None)

    par=par.parse_args()

    args = vars(par)
    for k, v in args.items():
        print(f"{k}:{v}")

    args['strategy'] = 'fsdp_native'
    return args

def train_foundation_model(args):
    logger = TensorBoardLogger(save_dir=args['log_dir'], name=args['run_name'])

    train_ds = PileH5Dataset(args['datapath'], 'train')
    val_ds = PileH5Dataset(args['datapath'], 'validation')

    traindl = DataLoader(train_ds, batch_size=2, num_workers=0)
    valdl = DataLoader(val_ds, batch_size=2, num_workers=2)

    if args['strategy'] == 'fsdp_native':
        strat_args = DDPFullyShardedNativeStrategy(
            cpu_offload=CPUOffload(offload_params=True),
            sharding_strategy= None,
            auto_wrap_policy= None,
            backward_prefetch=None,
            forward_prefetch=None,
        )
    model = LightningGPT(args)
    callbatcks = [EarlyStopping(monitor="val_step_loss", mode="min")]
    trainer = ptl.Trainer(
        logger=logger,
        strategy=strat_args,
        accelerator='gpu',
        devices=args['num_gpus']
        max_epochs=2,
        default_root_dir=args['log_dir'],
        limit_val_batches=1.0,
        limit_train_batches=1.0
    )

    trainer.fit(model, traindl, valdl, ckpt_path=None)

if __name__=='__main__':
    args = parse_args()
    train_foundation_model(args)