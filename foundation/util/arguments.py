from argparse import ArgumentParser as ap
"""
    Argument parser for LLMs using pytorch FSDP.
    We return args as a dict for easy integration into,
    e.g. ray[tune]
"""
def parse_bool(arg):
    # simply return true or false depending on input arg
    arg = arg.lower()
    if arg == 'true' or arg == 't':
        return True
    elif arg == 'false' or arg == 'f':
        return False
    else:
        raise NotImplemented(f"{arg} has no meaning!")
def parse_arguments():
    par = ap()

    par.add_argument('--num-gpus', '-gpus', type=int, default=None)

    par.add_argument('--model', type=str, 
                        default=None, 
                        help="Model to process", 
                        choices=['nanogpt'])
    par.add_argument('--time_file', type=str, default=None,
                     help='Lightning GPT only.  noop for pytorch_gpt')
    par.add_argument('--walltime', type=float, default=1)
    par.add_argument('--learn_rate', type=float, default=None)
    
    par.add_argument('--num_layers','-nl', type=int, default=12)
    par.add_argument('--num_heads', '-nh', type=int, default=12)
    par.add_argument('--embed_dim', '-ed', type=int, default=768)
    par.add_argument('--dropout', type=float, default=0.0,
                    help="do not use with Pytorch >= 2.0 and flash attention.  Which I've been told makes GPU go brrrrrr")

    par.add_argument('--local_rank', type=int, default=None)
    par.add_argument('--activation_checkpointing', '-ac', type=parse_bool, default='false',
                    help="checkpointing saves ~45% GPU memory.  I'd recommend it unless you have a good reason to not use it")
    par.add_argument('--cpu_offload', type=parse_bool, default='false', 
                    help="2/27/2023: offload + meta init + FSDP sharding is a logistical nightmare that doesn't seem to work")
    par.add_argument('--compile_model', type=parse_bool, default='false',
                    help="Currently crashes (2/27/2023) because of FSDP complications")
    
    
    par.add_argument('--max_epochs', type=int, default=5)
    par.add_argument('--batch_size', type=int, default=1)
    par.add_argument('--num_train_iter', type=int, default=None)
    par.add_argument('--num_val_iter', type=int, default=None)

    par.add_argument('--run_location', '-location', 
                                type=str, 
                                default=None, 
                                help="where are we running? at home or on a remote (sc)? Will skip .distributed stuff at home", 
                                choices=['home','sc', 'polaris'])
    par.add_argument('--log_dir', type=str, default='./')
    par.add_argument('--run_name', type=str, default=None)
    par.add_argument('--use_hdf5', type=parse_bool, default='False')
    par.add_argument('--datapath', type=str, default=None)
    par.add_argument('--training_files', type=str, nargs='*', default=None)
    par.add_argument('--validation_files', type=str, nargs='*', default=None)
    par.add_argument('--restart_ckpts', type=parse_bool, default='False', 
                    help="Should we output checkpoints when walltime runs low? requires also setting --walltime parameter")
    par.add_argument('--record_ckpts', type=parse_bool, default='False',
                    help="should we save a checkpoint when a record validation loss is achieved?")
    par.add_argument('--strategy', type=str, default=None)
    par.add_argument('--seq_length', type=int, default=2048)
    par.add_argument('--task', type=str, default='mask_gen', choices=['mask_gen', 'next_token'])
    
    par=par.parse_args()

    args = vars(par)
    for k, v in args.items():
        print(f"{k}:{v}")

    args['strategy'] = 'fsdp_native'
    return args