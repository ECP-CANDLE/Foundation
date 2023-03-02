import time
import torch.cuda as tcuda
import os, sys, shutil, json
from torch.utils.tensorboard import SummaryWriter

class LogAndProfiler():
    def __init__(self, save_log_filename, rank, save_format='json'):
        self.rank = rank
        self.save_fname = save_log_filename
        if not os.path.exists(self.save_fname):
            os.makedirs(self.save_fname, exist_ok=True)
        self.profiles = {}
        self.current_timers = {}
        self.writer = SummaryWriter(f"{save_log_filename}/runs/writer")
        self.init_time = time.time()


    def log_quantity(self, key, value):
        self.profiles[key] = value
    def start(self, label):
        self.current_timers[label] = time.time()

    def finish(self, label):
        if label not in self.current_timers.keys():
            print(f"Key {label} not found in current timers")
            raise KeyError
        if label not in self.profiles.keys():
            self.profiles[label] = []
        self.profiles[label].append(time.time() - self.current_timers[label])
        self.current_timers.pop(label)


    def tboard_log_scalar(self, label, value, num):
        self.writer.add_scalar(label, value, num)


    def log_quantity(self, label, value):
        if label not in self.profiles.keys():
            self.profiles[label] = []
        self.profiles[label].append(value)

    def save_log(self):
        with open(f"{self.save_fname}/rank{self.rank:03d}_profiles.json", 'w') as f:
            json.dump(self.profiles, f, indent=4)
    
    def log_cuda_memory(self, label):
        cm = tcuda.memory_stats()

        self.log_quantity(f'{label}/cuda_memory_active_peak', cm['active_bytes.all.peak']/1024**3)
        self.log_quantity(f'{label}/cuda_memory_allocated_peak', cm['allocated_bytes.all.peak']/1024**3)
        self.log_quantity(f'{label}/cuda_alloc_retries', cm['num_alloc_retries'])
        self.log_quantity(f"{label}/cuda_utilization", tcuda.utilization())
    def save_hyperparams(self, args):
        with open(f"{self.save_fname}/hparams.json", 'w') as f:
            json.dump(args, f, indent=4)

    def save_codebase(self):
        fname = f"{self.save_fname}"
        cpath = os.path.dirname(os.getcwd())
        # we assume that were running in the Foundation/foundation directory here
        if not os.path.exists(f"{fname}/codebase"):
            os.makedirs(f"{fname}/codebase", exist_ok=True)

        shutil.copytree(f"{cpath}/foundation/util", f"{fname}/codebase/util", dirs_exist_ok=True)
        shutil.copytree(f"{cpath}/foundation/models", f"{fname}/codebase/models", dirs_exist_ok=True)
        shutil.copy(f"{cpath}/foundation/pytorch_gpt.py", f"{fname}/codebase")

    def check_runtime(self):
        return (time.time() - self.init_time)/3600 # return hours