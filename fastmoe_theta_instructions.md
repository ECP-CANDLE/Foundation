# Environment setup
## Installing fastmoe
1. checked out fastmoe github from login node
2. login to thetagpusn1 & load modules
```
module load conda/2021-06-26
module load openmpi/openmpi-4.1.0_ucx-1.11.0_gcc-9.3.0
module load nccl/nccl-v2.9.9-1_CUDA11.3
```
3. create a conda env by cloning base env
4. build script modification
add nccl paths in `setup.py`. at line 62
```
include_dirs=['/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.9.9-1_gcc-9.3.0-17ubuntu1-20.04/include/'],
library_dirs=['/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.9.9-1_gcc-9.3.0-17ubuntu1-20.04/lib/'],
```
7. install
```
python setup.py install
```

# Prep Running
1. checking megatron (tag: v2.2) and apply [changes](https://github.com/laekov/fastmoe/blob/master/examples/megatron/fmoefy-v2.2.patch)
2. Note: do not apply `clip-grad-v2.2.patch` this caused hang
3. add barrier on `megatron/initialize.py`. at line 133
```
torch.distributed.barrier(device_ids=[args.local_rank])
```
4. cobalt script
```
#!/bin/bash -l
#COBALT -n 4
#COBALT -t 6:00:00
#COBALT -A CSC249ADOA01
#COBALT -M hsyoo@anl.gov

#export NCCL_DEBUG=INFO

NNODES=4
MASTER=`/bin/hostname -s`
echo "Master node: $MASTER"
WORKERS=''

while IFS= read line
do
  # echo "$line"
  if [ "$line" != "$MASTER" ]; then
      WORKERS+="$line "
  fi
done < $COBALT_NODEFILE

echo "Worker nodes: $WORKERS"

RANK=0

# Run workers
RANK=$((RANK+1))
for node in $WORKERS; do
    echo "ssh to $node"
    ssh -q $node /projects/CSC249ADOA01/hsyoo/fastmoe/Megatron-LM/run.sh "$NNODES" "$RANK" "$MASTER" &
    RANK=$((RANK+1))
done

# Run master
/projects/CSC249ADOA01/hsyoo/fastmoe/Megatron-LM/run.sh "$NNODES" 0 "$MASTER"
wait
```
6. run script
```
#!/bin/bash -l
set -e

# module load
module load conda/2021-06-26
module load openmpi/openmpi-4.1.0_ucx-1.11.0_gcc-9.3.0
module load nccl/nccl-v2.9.9-1_CUDA11.3

conda activate /projects/CSC249ADOA01/hsyoo/fastmoe/conda_env

cd /projects/CSC249ADOA01/hsyoo/fastmoe/Megatron-LM

NNODES=${1:-2}
RANK=${2:-0}
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MASTER_ADDR=${3:-localhost}
MASTER_PORT=6000

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$NNODES*$GPUS_PER_NODE))

DATA_PATH=/projects/CSC249ADOA01/hsyoo/fastmoe/gpt-wiki/my-gpt2_text_document
CHECKPOINT_PATH=/projects/CSC249ADOA01/hsyoo/fastmoe/workdir

python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_gpt.py \
        --fmoefy --num-experts 16 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --train-iters 500000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file gpt2-vocab.json \
        --merge-file gpt2-merges.txt \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --checkpoint-activations \
        --log-interval 100 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --fp16
```

