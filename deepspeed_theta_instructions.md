# Environmen setup
check `deepspeed_moe_theta_instructions.md` 

# Prep Running
Check `/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/` in theta, for the existing settings.

## pretrain.sh
Very similar to the MoE pretrain.sh. `run.sh` builds ds configs. this points to the different `node.sh` 
```
$ diff pretrain.sh examples/MoE/pretrain.sh
34c34,36
< ./run.sh
---
> bash ds_pretrain_gpt_1.3B_MoE128.sh
> # bash ds_pretrain_gpt_52B.sh # 52B total
> # bash ds_pretrain_gpt_6.7B_dense.sh
41,42c43,44
<     echo "ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh \"$ENCODED_WORLD\" $MASTER $RANK &"
<     ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh "$ENCODED_WORLD" $MASTER $RANK &
---
>     echo "ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/examples/MoE/node.sh \"$ENCODED_WORLD\" $MASTER $RANK &"
>     ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/examples/MoE/node.sh "$ENCODED_WORLD" $MASTER $RANK &
47c49
< /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh "$ENCODED_WORLD" $MASTER 0
---
> /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/examples/MoE/node.sh "$ENCODED_WORLD" $MASTER 0
```
Here is the full script.
```
#!/bin/bash -l

module load conda/2021-11-30
module load openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0
module load nccl/nccl-v2.11.4-1_CUDA11.4

conda activate /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/conda_env

export TORCH_EXTENSIONS_DIR=/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/pytorch_extensions/
export CUDA_LAUNCH_BLOCKING=1

mapfile -t </projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/exec_params.txt

WORLD_INFO=${1}
MASTER_ADDR=${2}
MASTER_PORT=29500
RANK=${3:-0}

if [ $RANK -eq 0 ]; then
    nvidia-smi -l 3 --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used -i 0 --format=csv > $COBALT_JOBID.log &
    MONITOR=$!
    echo "Monitor PID: $MONITOR"
fi

run_cmd="/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/conda_env/bin/python -u -m deepspeed.launcher.launch --world_info=$WORLD_INFO --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${MAPFILE[@]}"

eval ${run_cmd}

if [ $RANK -eq 0 ]; then
    echo "Terminating monitor process $MONITOR"
    kill -9 $MONITOR
fi
hsyoo@thetagpusn2:/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed$ cat pretrain
cat: pretrain: No such file or directory
hsyoo@thetagpusn2:/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed$ cat pretrain.sh
#!/bin/bash -l
#COBALT -n 8
#COBALT -t 00:30:00
#COBALT -A CSC249ADOA01
#COBALT -M hsyoo@anl.gov

MASTER=`/bin/hostname -s`
echo "Master node: $MASTER"

# collect worker nodes
WORKERS=''
while IFS= read line
do
  # echo "$line"
  if [ "$line" != "$MASTER" ]; then
      WORKERS+="$line "
  fi
done < $COBALT_NODEFILE
echo "Worker nodes: $WORKERS"


# compose WORLD info
WORLD=''
for WORKER in $WORKERS;do
  if [ ${#WORLD} -ne 0 ]; then
    WORLD="$WORLD,"
  fi
  WORLD="$WORLD\"$WORKER\":[0,1,2,3,4,5,6,7]"
done
WORLD="{$WORLD,\"$MASTER\":[0,1,2,3,4,5,6,7]}"
ENCODED_WORLD=`echo $WORLD | /usr/bin/base64 -w 0`
echo "WORLD: $WORLD, ENCODED: $ENCODED_WORLD"

./run.sh

# Run workers
RANK=0
RANK=$((RANK+1))
for node in $WORKERS; do
    echo "ssh to $node"
    echo "ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh \"$ENCODED_WORLD\" $MASTER $RANK &"
    ssh -q $node /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh "$ENCODED_WORLD" $MASTER $RANK &
    RANK=$((RANK+1))
done

# Run master
/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/node.sh "$ENCODED_WORLD" $MASTER 0
wait
```

## run.sh
```
#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

BASE_DATA_PATH=/projects/CSC249ADOA01/hsyoo/fastmoe/gpt-wiki
DATASET=${BASE_DATA_PATH}/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=1


# Debug
#TP=4
#PP=4
#LAYERS=8
#HIDDEN=512
#SEQ=1024
#GLOBAL_BATCH=128
#WORKER_STR="-i worker-0"


# 1.7B
# 3.6B
# 7.5B
# 18B
TP=8
PP=1
HIDDEN=6144
LAYERS=40
SEQ=1024
MICRO_BATCH=16
GLOBAL_BATCH=512
WORKER_STR=""


while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads 32 \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters 10 \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --fp16 \
	--checkpoint-activations
        "


if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi


cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": false,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"
run_cmd="${DIR}/pretrain_gpt.py $@ ${options}"


echo ${run_cmd} > "exec_params.txt"
# eval ${run_cmd}

set +x
```

## node.sh
Very similar to the MoE node.sh. Just points to different path for the ds config.
```
diff node.sh examples/MoE/node.sh
10a11
> # export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
12c13
< mapfile -t </projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/exec_params.txt
---
> mapfile -t </projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/examples/MoE/exec_params.txt
```
Here is a full script.
```
#!/bin/bash -l

module load conda/2021-11-30
module load openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0
module load nccl/nccl-v2.11.4-1_CUDA11.4

conda activate /projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/conda_env

export TORCH_EXTENSIONS_DIR=/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/pytorch_extensions/
export CUDA_LAUNCH_BLOCKING=1

mapfile -t </projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/Megatron-DeepSpeed/exec_params.txt

WORLD_INFO=${1}
MASTER_ADDR=${2}
MASTER_PORT=29500
RANK=${3:-0}

if [ $RANK -eq 0 ]; then
    nvidia-smi -l 3 --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used -i 0 --format=csv > $COBALT_JOBID.log &
    MONITOR=$!
    echo "Monitor PID: $MONITOR"
fi

run_cmd="/projects/CSC249ADOA01/hsyoo/DS_MoE_NLG/conda_env/bin/python -u -m deepspeed.launcher.launch --world_info=$WORLD_INFO --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${MAPFILE[@]}"

eval ${run_cmd}

if [ $RANK -eq 0 ]; then
    echo "Terminating monitor process $MONITOR"
    kill -9 $MONITOR
fi

```