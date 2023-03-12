# Development notes on Foundation

## 50B parameter scaling tests and comparison tests

### Comparison tests:
Comparing DeepSpeed, ColossalAI, and FSDP.  The purpose is to see how FSDP compares to these more complicated models.  In order to achieve best-to-best comparison, we have slight environment changes between the runs

* DS and CAI use Pytorch 1.13 with the built-in conda environment on polaris; cloned from conda/2022-09-08
* FSDP uses Pytorch 2.0 with scratch conda installation initially cloned from conda/2023-01-10-unstable

Use GPT2 from huggingface with configurations:
```
vocab_size = 50304
block_size = 2048
n_layer = 64
n_head = 64
n_embd = 8192
n_inner = 4 * 8192
activation_function = 'gelu'
dropout = 0
```

Deepspeed:  
* 10 nodes with cpu offload and activation checkpointing (not sure if checkpointing does anything on HF models... but it seems to since
GPU usage is similar to FSDP)


## Bug Notes
### pytorch 2.0 + deepspeed
Running DS gives an error:
```
    util_ops = UtilsBuilder().load()
        TypeError: 'NoneType' object is not callable
```
linked to https://github.com/microsoft/DeepSpeed/issues/2904.

Need to update the imports (deepspeed version 0.8.2) as here:
https://github.com/microsoft/DeepSpeed/pull/2943/files