import torch, os
from functools import partial
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies import ColossalAIStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                BackwardPrefetch,
                                                                StateDictType
)
from foundation.models.minGPT import Block

def setup_environment(args, machine):
    if machine == 'polaris':
        os.environ['RANK'] = os.environ['PMI_RANK']# global 
        os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
        args['master_addr'] = os.environ['MASTER_ADDR']
        args['world_size'] = int(os.environ['PMI_SIZE'])
        args['global_rank'] = int(os.environ['PMI_RANK'])
        args['local_rank'] = int(os.environ['PMI_LOCAL_RANK']) # wraparound since LOCAL_RANK is actually global?? WRT/ 
        args['local_size'] = int(os.environ['PMI_LOCAL_SIZE'])
        args['backend'] = 'nccl'
        args['node_id'] = args['global_rank'] // args['local_size']
        args['num_nodes'] = args['world_size'] // args['local_size']
    else:
        print(f"{machine} not recognized, cannot initialize distributed environment")
        exit()
    return args
            
class PolarisEnvironment(ClusterEnvironment):
    def __init__(self):
        super(PolarisEnvironment, self).__init__()

    @property
    def creates_processes_externally(self) -> bool:
        return True
    def detect(self) -> bool:
        return "PMI_SIZE" in os.environ.keys()
    def world_size(self) -> int:
        return int(os.environ['PMI_SIZE'])
    def global_rank(self) -> int:
        return int(os.environ['PMI_RANK'])
    def local_rank(self) -> int:
        return int(os.environ['PMI_LOCAL_RANK'])
    def node_rank(self) -> int:
        return int(os.environ['PMI_RANK'])//int(os.environ['PMI_LOCAL_SIZE'])
    @property
    def main_address(self) -> str:
        return os.environ['MASTER_ADDR']
    @property
    def main_port(self) -> int:
        return int(os.environ['MASTER_PORT'])
    def set_global_rank(self, rank: int = 0) -> None:
        pass
    def set_world_size(self, size: int = 1) -> None:
        pass

def setup_strategy(args: dict, model):

    if args['strategy'] == 'fsdp_native':
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block,}
        )

        strat_args = DDPFullyShardedNativeStrategy(
            cluster_environment=PolarisEnvironment(),
            parallel_devices=[torch.device('cuda:%d'%d) for d in [0,1,2,3]]*args['num_nodes'],          
            cpu_offload=CPUOffload(offload_params=args['cpu_offload']),
            sharding_strategy= ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy= twrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            process_group_backend="nccl",
            activation_checkpointing=(Block,) if args['activation_checkpointing'] else None
        )

    elif args['strategy'] == 'deepspeed':
        strat_args= DeepSpeedStrategy(
            zero_optimization=True,
            stage=3,
            offload_optimizer=args['cpu_offload'],
            offload_parameters=args['cpu_offload'],
            remote_device='cpu',
            offload_params_device='cpu',
            offload_optimizer_device='cpu',
            nvme_path='/local/scratch',
            logging_batch_size_per_gpu=args['batch_size'],
            partition_activations=args['activation_checkpointing'],
            cpu_checkpointing=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            pin_memory=True,
            contiguous_memory_optimization=False
            # add the option to load a config from json file with more deepspeed options
            # note that if supplied all defaults are ignored - model settings defaults this arg to None
            # config=cfg.deepspeed_cfg_file
        )
    elif args['strategy'] == 'colossalai':
        strat_args='colossal_ai'

    return strat_args