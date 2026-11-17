"""Run with torchrun --standalone --nproc-per-node=2 to check NCCL collection."""
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.utils.comm import all_gather

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl')
try:
    rank = dist.get_rank()
    payload = {'rank': rank, 'values': list(range(rank * 100 + 1))}
    gathered = all_gather(payload)
    assert len(gathered) == dist.get_world_size()
    assert all(p['rank'] == i and len(p['values']) == i * 100 + 1 for i, p in enumerate(gathered))
    if rank == 0:
        print('NCCL variable-length prediction gathering passed', flush=True)
finally:
    dist.destroy_process_group()
