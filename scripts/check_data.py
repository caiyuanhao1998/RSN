"""Audit downloaded datasets and exercise real training/evaluation batches."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('RSN_HOME', str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['RSN18.coco', 'RSN18.mpii'], required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(ROOT / args.experiment))
from config import cfg
from network import RSN
from lib.utils.dataloader import get_train_loader, get_test_loader
import test as evaluation
from pycocotools.cocoeval import COCOeval

torch.set_num_threads(4)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_GPU = 2
cfg.SOLVER.MAX_ITER = 1
cfg.TEST.IMS_PER_GPU = 2
args.output.mkdir(parents=True, exist_ok=True)
report = dict(experiment=args.experiment, checkpoint=str(args.checkpoint))
train_loader = get_train_loader(cfg, 1, is_dist=False)
train_dataset = train_loader.dataset
missing = [p for p in {d['img_path'] for d in train_dataset.data} if not Path(p).is_file()]
report['training_people'] = len(train_dataset)
report['missing_train_images'] = len(missing)
assert not missing, missing[:5]
images, valids, labels = next(iter(train_loader))
report['train_shapes'] = [list(x.shape) for x in (images, valids, labels)]
assert all(torch.isfinite(x).all() for x in (images, valids, labels))
val_loader = get_test_loader(cfg, 1, 0, 'val', is_dist=False)
missing = [p for p in {d['img_path'] for d in val_loader.ori_dataset.data} if not Path(p).is_file()]
report['validation_people_or_detections'] = len(val_loader.ori_dataset)
report['missing_val_images'] = len(missing)
assert not missing, missing[:5]
# Three real examples exercise a final batch of one, including flip inference.
subset = Subset(val_loader.ori_dataset, [0, 1, 2])
loader = DataLoader(subset, batch_size=2, collate_fn=val_loader.collate_fn, num_workers=0)
model = RSN(cfg).cuda().eval()
loaded = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(loaded['model'])
results = evaluation.compute_on_dataset(model, loader, torch.device('cuda'))
if cfg.DATASET.NAME == 'COCO':
    assert len(results) == 3
    (args.output/'results.json').write_text(json.dumps(results))
    gt = val_loader.ori_dataset.val_gt
    pred = gt.loadRes(results)
    coco_eval = COCOeval(gt, pred, 'keypoints')
    coco_eval.params.imgIds = sorted({r['image_id'] for r in results})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    report['evaluation_images'] = [int(i) for i in coco_eval.params.imgIds]
    report['subset_AP'] = float(coco_eval.stats[0])
else:
    predictions = np.concatenate(results)
    assert predictions.shape == (3, 16, 3)
    assert np.isfinite(predictions).all()
    np.save(args.output/'predictions.npy', predictions)
    report['prediction_shape'] = list(predictions.shape)
report['note'] = 'Short runtime smoke test; metrics are not trained-model benchmark results.'
(args.output/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
