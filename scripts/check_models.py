"""Check each provided model's CUDA forward/backward and efficient mode."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('RSN_HOME', str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['Res18.coco', 'RSN18.coco', '4XRSN18.coco', 'RSN18.mpii'], required=True)
parser.add_argument('--efficient', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(ROOT / args.experiment))
from config import cfg
import network

torch.set_num_threads(4)
torch.manual_seed(42)
cls = getattr(network, 'RSN', None) or network.MSPN
model = cls(cfg, run_efficient=args.efficient).cuda().train()
images = torch.randn(2, 3, *cfg.INPUT_SHAPE, device='cuda')
valids = torch.full((2, cfg.DATASET.KEYPOINT.NUM, 1), 2., device='cuda')
labels = torch.randn(2, 5, cfg.DATASET.KEYPOINT.NUM, *cfg.OUTPUT_SHAPE, device='cuda')
loss = model(images, valids, labels)['total_loss']
assert torch.isfinite(loss)
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad]
assert all(g is not None and torch.isfinite(g).all() for g in grads)
model.eval()
with torch.no_grad():
    heatmaps = model(images[:1])
assert heatmaps.shape == (1, cfg.DATASET.KEYPOINT.NUM, *cfg.OUTPUT_SHAPE)
assert torch.isfinite(heatmaps).all()
print(json.dumps(dict(experiment=args.experiment, efficient=args.efficient, loss=loss.item(),
                      output_shape=list(heatmaps.shape), parameters=sum(p.numel() for p in model.parameters()),
                      peak_memory_MiB=round(torch.cuda.max_memory_allocated() / 2**20, 1))))
