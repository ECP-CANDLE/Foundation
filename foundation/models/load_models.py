
import torch
import deepspeed

from foundation.models.minGPT import (
        GPT, 
        GPTConfig, 
        Block, 
        MLP, 
        CausalSelfAttention
        )
from transformers import (
        GPTNeoXConfig, 
        GPT2Config
)     
from transformers import (
    GPTNeoXForCausalLM,
    GPT2LMHeadModel
)
tv = [int(torch.__version__.split('_')[0].split('.')[i]) for i in range(2)]
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXMLP
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
if hasattr(torch.distributed.algorithms._checkpoint.checkpoint_wrapper, "apply_activation_checkpointing"):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
    fsdp_ckpt = True
else:
    print("WARNING: torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing not found")
    print("FSDP will not apply activation checkpointing")
    fsdp_ckpt = False
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                BackwardPrefetch,
                                                                StateDictType
)
from functools import partial

def init_fsdp(arch, config, args, twrap_policy, mixed_precision):
    model = FSDP(arch(config),
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
    return model
def init_ds_model(args):
    if args.model == 'nanogpt':
        config = GPTConfig(
            block_size = args.seq_length, # configured in tokenizer to match GPT-3
            vocab_size = 50304,
            n_layer = args.num_layers,
            n_head = args.num_heads,
            n_embd = args.embed_dim,
            dropout = args.dropout,
            bias = False,
        )
        with deepspeed.zero.Init():
            model = GPT(config) 
    elif args.model == 'gpt2_hf':
        config = GPT2Config(
            vocab_size = 50304,
            n_positions = args['seq_length'],
            n_embd = args['embed_dim'],
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_inner = args['embed_dim']*4,
            activation_function = 'gelu',

        )
        with deepspeed.zero.Init():
            model = GPT2LMHeadModel(config)  

    elif args.model == 'gpt_neox':
        config = GPTNeoXConfig(
            vocab_size = 50304,
            n_positions = args['seq_length'],
            n_embd = args['embed_dim'],
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_inner = args['embed_dim']*4,
            activation_function = 'gelu',

        )
        with deepspeed.zero.Init():
            model = GPTNeoXForCausalLM(config)
    return model

def init_model(args):
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    if args['model'] == 'nanogpt':
        config = GPTConfig(
            block_size = args['seq_length'], # configured in tokenizer to match GPT-3
            vocab_size = 50304,
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_embd = args['embed_dim'],
            dropout = args['dropout'],
            bias = False,
        )
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block,}
        )
        if fsdp_ckpt:
            ckpt_fn = lambda submod: isinstance(submod, Block)
            non_reent_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=args['cpu_offload'],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
        model = init_fsdp(GPT, config, args, twrap_policy, mixed_precision)
    
    elif args['model'] == 'gpt2_hf':
        config = GPT2Config(
            vocab_size = 50304,
            n_positions = args['seq_length'],
            n_embd = args['embed_dim'],
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_inner = args['embed_dim']*4,
            activation_function = 'gelu',

        )
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPT2Block,}
        )
        if fsdp_ckpt:
            ckpt_fn = lambda submod: isinstance(submod, GPT2Block)
            non_reent_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=args['cpu_offload'],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
        model = init_fsdp(GPT2LMHeadModel, config, args, twrap_policy, mixed_precision)
            
            
    elif args['model'] == 'gpt_neox':
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
        twrap_policy = wrap_policy
        twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPTNeoXAttention, GPTNeoXMLP}
        )
        if fsdp_ckpt:
            ckpt_fn = lambda submod: isinstance(submod, GPTNeoXAttention) or isinstance(submod, GPTNeoXMLP)
            non_reent_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=args['cpu_offload'],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
        model = init_fsdp(GPTNeoXForCausalLM, config, args, twrap_policy, mixed_precision)



    if args['activation_checkpointing'] and fsdp_ckpt:
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reent_wrapper, check_fn=ckpt_fn
        )
        # print(f"Rank {args['global_rank']} finished AC wrapper in {time.time()-start} seconds...")
    if args['compile_model']:
        model = torch.compile(model)


    
    print(f"[{args['global_rank']}] initializing {args['model']} completed.")
    return model


def get_config(args: dict):
    init_dev = 'meta' if args['strategy'] == 'fsdp_native' or args['strategy'] == 'colossalai' else None

    if args['model'] == 'nanogpt':
        config = GPTConfig(
                    block_size = args['seq_length'], # configured in tokenizer to match GPT-3
                    vocab_size = 50304,
                    n_layer = args['num_layers'],
                    n_head = args['num_heads'],
                    n_embd = args['embed_dim'],
                    dropout = args['dropout'],
                    bias = True
        )
    elif args['model'] == 'gpt2_hf':
        config = GPT2Config(
            vocab_size = 50304,
            n_positions = args['seq_length'],
            n_embd = args['embed_dim'],
            n_layer = args['num_layers'],
            n_head = args['num_heads'],
            n_inner = args['embed_dim']*4,
            activation_function = 'gelu',
        )
    elif args['model'] == 'gpt_neox':
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
    return config