"""Fine-tune RSN + RTMPose-X using a single COCO + MPII training entry point."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from data_pipeline import (
    DATA, ROOT, WIDTH, FLIP, prepare_records, preprocess, merge_simcc_flip,
    simcc_loss, simcc_oks_loss, simcc_pose_oks_loss, pose_quality_loss,
    pose_oks_rank_loss, local_joint_loss, pose_proposal_loss,
)
from evaluate import distributed_setup, load_model, run_evaluation
from expand_refiner import capacity_candidate
from heatmap_supervision import heatmap_distribution_loss, heatmap_loss
from mixed_data import CocoMpiiPose, prepare_mpii
from models import RSN_PARAMETER_PREFIXES, RSNSpatialAttention, RTMPoseX
from retention import archive_ranked_epoch, checkpoint_lock, model_score


# Fixed handoff boundaries from the supervised training runs.
RELEASE_RECIPE = {'architecture': {'variant': 'rsn_highres',
                  'rsn_steps': 4,
                  'width': 128,
                  'lateral_channels': 160,
                  'spatial_stride': 8,
                  'context_width': 160,
                  'local_width': 96,
                  'trunk_width': 0,
                  'spatial_width': 24,
                  'pixel_width': 0,
                  'spatial_attention_width': 0,
                  'spatial_attention_layers': 1,
                  'spatial_rsn_blocks': 0,
                  'spatial_rsn_width': 0,
                  'spatial_rsn_attention_layers': 1,
                  'raw_confidence_output': True,
                  'pose_oks_confidence': False,
                  'proposal_confidence': False,
                  'proposal_size_confidence': False},
 'common': {'training_data': 'coco_mpii', 'amp': 'fp16', 'mpii_loss_weight': 0.75, 'seed': 21},
 'stages': [{'name': 'spatial_heatmaps',
             'effective_batch': 4096,
             'handoff_epoch': 12,
             'handoff_updates': 504,
             'options': {'epochs': 60,
                         'batch_size': 256,
                         'lr': 2e-05,
                         'backbone_lr': 8e-06,
                         'new_lr': 0.00018,
                         'context_lr': 0.00022,
                         'warmup_updates': 84,
                         'checkpoint_refiner': True,
                         'freeze_base': True,
                         'train_backbone_from': 3,
                         'heatmap_loss_weight': 0.08,
                         'heatmap_sigma': 1.5,
                         'oks_loss_weight': 0.12,
                         'quality_loss_weight': 0.1,
                         'local_loss_weight': 0.06,
                         'bbox_transform': 'official',
                         'schedule_updates': 1680}},
            {'name': 'wide_spatial_head',
             'effective_batch': 4096,
             'handoff_epoch': 15,
             'handoff_updates': 630,
             'architecture': {'width': 136, 'lateral_channels': 168, 'context_width': 168},
             'options': {'epochs': 50,
                         'batch_size': 256,
                         'lr': 6e-06,
                         'backbone_lr': 6e-06,
                         'new_lr': 8e-05,
                         'context_lr': 9e-05,
                         'spatial_gain_lr': 0.0006,
                         'warmup_updates': 42,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'heatmap_loss_weight': 0.04,
                         'heatmap_sigma': 1.25,
                         'oks_loss_weight': 0.15,
                         'quality_loss_weight': 0.1,
                         'local_loss_weight': 0.06,
                         'bbox_transform': 'official',
                         'bbox_shift_factor': 0.1,
                         'simcc_sigma_scale': 0.96,
                         'simcc_continuous': True,
                         'simcc_hard_joint_weight': 0.1,
                         'schedule_updates': 1512}},
            {'name': 'proposal_confidence',
             'effective_batch': 4096,
             'handoff_epoch': 8,
             'handoff_updates': 336,
             'architecture': {'raw_confidence_output': False,
                              'pose_oks_confidence': True,
                              'proposal_confidence': True},
             'options': {'epochs': 12,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.0006,
                         'warmup_updates': 21,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12,
                         'batch_size': 512}},
            {'name': 'hard_proposals',
             'effective_batch': 4096,
             'handoff_epoch': 2,
             'handoff_updates': 84,
             'architecture': {},
             'options': {'epochs': 16,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.00025,
                         'warmup_updates': 21,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12,
                         'batch_size': 512}},
            {'name': 'proposal_size',
             'effective_batch': 4096,
             'handoff_epoch': 5,
             'handoff_updates': 210,
             'architecture': {'proposal_size_confidence': True},
             'options': {'epochs': 12,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.0008,
                         'warmup_updates': 21,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'proposal_size_only': True,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12,
                         'batch_size': 512}},
            {'name': 'joint_proposal_confidence',
             'effective_batch': 4096,
             'handoff_epoch': 12,
             'handoff_updates': 504,
             'architecture': {},
             'options': {'epochs': 12,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.00012,
                         'warmup_updates': 21,
                         'inherit_ema_age': True,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12,
                         'batch_size': 512}},
            {'name': 'joint_geometry',
             'effective_batch': 12288,
             'handoff_epoch': 2,
             'handoff_updates': 28,
             'architecture': {},
             'options': {'lr': 3e-06,
                         'backbone_lr': 2e-06,
                         'new_lr': 4e-05,
                         'spatial_gain_lr': 0.0002,
                         'warmup_updates': 14,
                         'inherit_ema_age': True,
                         'checkpoint_refiner': True,
                         'freeze_refiner_bn': True,
                         'initial_weight_anchor': 150.0,
                         'train_backbone_from': 5,
                         'heatmap_loss_weight': 0.04,
                         'heatmap_sigma': 1.25,
                         'oks_loss_weight': 0.12,
                         'pose_oks_localization_weight': 0.75,
                         'quality_loss_weight': 0.0,
                         'joint_refinement_only': True,
                         'local_loss_weight': 0.1,
                         'bbox_transform': 'official',
                         'bbox_shift_factor': 0.08,
                         'simcc_sigma_scale': 0.96,
                         'simcc_continuous': True,
                         'simcc_hard_joint_weight': 0.1,
                         'batch_size': 512,
                         'epochs': 20}},
            {'name': 'wider_spatial_geometry',
             'effective_batch': 12288,
             'handoff_epoch': 1,
             'handoff_updates': 14,
             'architecture': {'spatial_width': 128},
             'options': {'epochs': 6,
                         'batch_size': 256,
                         'lr': 3e-06,
                         'backbone_lr': 2e-06,
                         'new_lr': 6.5e-05,
                         'spatial_gain_lr': 0.0002,
                         'warmup_updates': 7,
                         'checkpoint_refiner': True,
                         'freeze_refiner_bn': True,
                         'initial_weight_anchor': 150.0,
                         'train_backbone_from': 5,
                         'heatmap_loss_weight': 0.04,
                         'heatmap_sigma': 1.25,
                         'oks_loss_weight': 0.12,
                         'pose_oks_localization_weight': 0.75,
                         'quality_loss_weight': 0.0,
                         'joint_refinement_only': True,
                         'local_loss_weight': 0.1,
                         'bbox_transform': 'official',
                         'bbox_shift_factor': 0.08,
                         'simcc_sigma_scale': 0.96,
                         'simcc_continuous': True,
                         'simcc_hard_joint_weight': 0.1}},
            {'name': 'wider_spatial_confidence',
             'effective_batch': 12288,
             'handoff_epoch': 8,
             'handoff_updates': 112,
             'architecture': {},
             'options': {'epochs': 8,
                         'batch_size': 768,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.0001,
                         'warmup_updates': 9,
                         'inherit_ema_age': True,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12}},
            {'name': 'spatial_attention_warmup',
             'effective_batch': 12288,
             'handoff_epoch': 3,
             'handoff_updates': 42,
             'architecture': {'spatial_width': 192, 'spatial_attention_width': 96},
             'options': {'epochs': 6,
                         'lr': 3e-06,
                         'backbone_lr': 2e-06,
                         'new_lr': 0.00018,
                         'warmup_updates': 7,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'heatmap_loss_weight': 0.05,
                         'heatmap_sigma': 1.25,
                         'oks_loss_weight': 0.15,
                         'pose_oks_localization_weight': 0.75,
                         'quality_loss_weight': 0.0,
                         'spatial_attention_only': True,
                         'bbox_transform': 'official',
                         'bbox_shift_factor': 0.08,
                         'simcc_sigma_scale': 0.96,
                         'simcc_continuous': True,
                         'simcc_hard_joint_weight': 0.1,
                         'batch_size': 512}},
            {'name': 'spatial_attention_geometry',
             'effective_batch': 12288,
             'handoff_epoch': 3,
             'handoff_updates': 42,
             'architecture': {},
             'options': {'epochs': 14,
                         'batch_size': 256,
                         'lr': 3e-06,
                         'backbone_lr': 2e-06,
                         'new_lr': 6.5e-05,
                         'spatial_gain_lr': 0.0002,
                         'warmup_updates': 7,
                         'checkpoint_refiner': True,
                         'freeze_refiner_bn': True,
                         'initial_weight_anchor': 150.0,
                         'train_backbone_from': 5,
                         'heatmap_loss_weight': 0.04,
                         'heatmap_sigma': 1.25,
                         'oks_loss_weight': 0.12,
                         'pose_oks_localization_weight': 0.75,
                         'quality_loss_weight': 0.0,
                         'joint_refinement_only': True,
                         'local_loss_weight': 0.1,
                         'bbox_transform': 'official',
                         'bbox_shift_factor': 0.08,
                         'simcc_sigma_scale': 0.96,
                         'simcc_continuous': True,
                         'simcc_hard_joint_weight': 0.1}},
            {'name': 'spatial_attention_confidence',
             'effective_batch': 12288,
             'handoff_epoch': 8,
             'handoff_updates': 112,
             'architecture': {},
             'options': {'epochs': 8,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 0.0001,
                         'warmup_updates': 9,
                         'inherit_ema_age': True,
                         'checkpoint_refiner': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12,
                         'batch_size': 512}},
            {'name': 'final_simcc_confidence',
             'effective_batch': 14336,
             'handoff_epoch': 6,
             'handoff_updates': 72,
             'architecture': {},
             'options': {'epochs': 6,
                         'batch_size': 448,
                         'lr': 1e-05,
                         'backbone_lr': 4e-06,
                         'new_lr': 6.5e-05,
                         'warmup_updates': 6,
                         'inherit_ema_age': True,
                         'train_backbone_from': 5,
                         'quality_loss_weight': 1.0,
                         'object_proposals': 'coco_train_object_proposals.npz',
                         'pose_score_only': True,
                         'augment': 'pose_quality',
                         'bbox_shift_factor': 0.12}}]}


def canonical_metadata(data):
    # Remove str subclasses such as torch.torch_version.TorchVersion.
    # Plain JSON-compatible metadata round-trips through restricted torch.load.
    return json.loads(json.dumps(data))


def make_optimizer(model, args):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = args.new_lr if name.startswith(RSN_PARAMETER_PREFIXES) else (
            args.backbone_lr if name.startswith('backbone.') else args.lr)
        if getattr(args, 'context_lr', None) is not None and name.startswith(
                ('head.refiner.context_steps.', 'head.refiner.axis_x.', 'head.refiner.axis_y.',
                 'head.refiner.quality.', 'head.refiner.local.', 'head.refiner.spatial.', 'head.refiner.pixel.')):
            lr = args.context_lr
        if getattr(args, 'spatial_gain_lr', None) is not None and name == 'head.refiner.spatial.axis_gain':
            lr = args.spatial_gain_lr
        if name.startswith('backbone.') and not name.startswith(RSN_PARAMETER_PREFIXES):
            layer_decay = getattr(args, 'backbone_layer_decay', 1.)
            stage = next((i for i in range(1, 5) if name.startswith(f'backbone.stage{i}.')), 0)
            lr *= layer_decay**(4-stage)
        decay = 0. if parameter.ndim < 2 or name.endswith('.bias') else args.weight_decay
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=lr, initial_lr=lr, weight_decay=decay)
                              for (lr, decay), params in groups.items()], fused=True)


def seed_worker(worker):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


def ema_pairs(ema, model):
    pairs = list(zip(ema.state_dict().values(), model.state_dict().values()))
    floating = [(a, b) for a, b in pairs if a.is_floating_point()]
    integral = [(a, b) for a, b in pairs if not a.is_floating_point()]
    return [a for a, _ in floating], [b for _, b in floating], integral


@torch.no_grad()
def update_ema(pairs, decay):
    targets, sources, integral = pairs
    torch._foreach_lerp_(targets, sources, 1-decay)
    for a, b in integral: a.copy_(b)


def cpu_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record_session(output, settings, source_manifest, start_epoch, updates):
    resume = settings.get('resume')
    row = dict(started_utc=datetime.now(timezone.utc).isoformat(), completed_epochs=start_epoch,
               completed_updates=updates, settings=settings,
               source_manifest_sha256=file_sha256(source_manifest),
               initialization_sha256=file_sha256(settings['checkpoint']),
               resume_sha256=file_sha256(resume) if resume else None)
    with (output/'training_sessions.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def records(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may still be appending its final line.
            continue
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


OFFICIAL_SHA256 = '71d7b7e957646a52c504bdaff511dfa37813f99b7725378c5530ba6eeaaab200'
COCO_ARCHIVE_URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
DETECTIONS = 'COCO_val2017_detections_AP_H_56_person.json'


def prepare_objects(instances, train_ids, archive_sha256, output):
    if train_ids != {row['id'] for row in instances['images']}:
        raise ValueError('COCO instance and keypoint training image IDs differ')
    people = defaultdict(list)
    for row in instances['annotations']:
        if row['category_id'] == 1 and min(row['bbox'][2:]) > 0:
            people[row['image_id']].append(row['bbox'])
    people = {image: np.asarray(boxes, dtype=np.float32) for image, boxes in people.items()}
    image_ids, boxes, categories = [], [], []
    for row in instances['annotations']:
        if row['category_id'] == 1 or row['iscrowd'] or min(row['bbox'][2:]) < 8:
            continue
        image = row['image_id']
        if image not in train_ids:
            raise ValueError(f'Object annotation refers to a non-training image: {image}')
        refs = people.get(image)
        if refs is not None:
            x, y, w, h = row['bbox']
            cx, cy = x+w*.5, y+h*.5
            cw, ch = 1.25*max(w, h*.75), 1.25*max(h, w/.75)
            lo = np.array([cx-cw*.5, cy-ch*.5])
            hi = np.array([cx+cw*.5, cy+ch*.5])
            overlap = np.maximum(np.minimum(hi, refs[:, :2]+refs[:, 2:]) -
                                 np.maximum(lo, refs[:, :2]), 0.).prod(1)
            if ((overlap > .03*cw*ch) | (overlap > .03*refs[:, 2:].prod(1))).any():
                continue
        image_ids.append(image)
        boxes.append(row['bbox'])
        categories.append(row['category_id'])
    if not boxes:
        raise ValueError('The training set yielded no non-person object proposals')
    np.savez_compressed(output, image_ids=np.asarray(image_ids, dtype=np.int64),
                        boxes=np.asarray(boxes, dtype=np.float32),
                        category_ids=np.asarray(categories, dtype=np.int16),
                        annotation_archive_sha256=archive_sha256, annotation_archive_url=COCO_ARCHIVE_URL)
    return len(boxes)


def prepare_data(args):
    destination = args.output.expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'{destination}: choose an empty directory for the prepared dataset')
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_sha256 = file_sha256(args.coco_archive)
    with TemporaryDirectory(prefix='rsn_prepare_', dir=destination.parent) as scratch:
        cache = Path(scratch)/'data'
        (cache/'annotations').mkdir(parents=True)
        with ZipFile(args.coco_archive) as archive:
            for split in ('train', 'val'):
                entry = f'annotations/person_keypoints_{split}2017.json'
                with archive.open(entry) as source, (cache/entry).open('wb') as target:
                    shutil.copyfileobj(source, target)
            shutil.copyfile(args.detections, cache/DETECTIONS)
            prepare_records(args.coco_images, target=cache)
            prepare_mpii(args.mpii_annotations, args.mpii_images, target=cache)
            train = json.loads((cache/'annotations/person_keypoints_train2017.json').read_text())
            train_ids = {row['id'] for row in train['images']}
            del train
            with archive.open('annotations/instances_train2017.json') as source:
                instances = json.load(source)
            objects = prepare_objects(instances, train_ids, archive_sha256, cache/'coco_train_object_proposals.npz')
        report = dict(coco_archive_sha256=archive_sha256, coco_detections_sha256=file_sha256(args.detections),
                      mpii_annotations_sha256=file_sha256(args.mpii_annotations),
                      coco_train_object_proposals=objects)
        (cache/'release_data.json').write_text(json.dumps(report, indent=2)+'\n')
        cache.replace(destination)
    print(json.dumps(report, indent=2))


def convert_official(source, output):
    if file_sha256(source) != OFFICIAL_SHA256:
        raise ValueError('The official RTMPose-X checkpoint SHA-256 does not match the recorded source')
    # This exact upstream checkpoint includes numeric NumPy metadata in legacy pickle form.
    numeric = [type(np.dtype(name)) for name in ('float32', 'float64', 'int32', 'int64', 'uint8')]
    allowed = [(np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype, *numeric]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(source, map_location='cpu', weights_only=True)['state_dict']
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor-only state dictionary')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(state_dict=state), output)
    print(json.dumps(dict(checkpoint=str(output), sha256=file_sha256(output), tensors=len(state))))


def configure_trainable(model, freeze_base, train_backbone_from=0, pose_score_only=False,
                        pixel_refinement_only=False, proposal_size_only=False, freeze_pose_confidence=False,
                        joint_refinement_only=False, spatial_attention_only=False, coarse_attention_only=False,
                        simcc_output_only=False, spatial_rsn_blocks_only=False, train_spatial_axis_gain=False):
    if not 0 <= train_backbone_from <= 5:
        raise ValueError('Backbone stage must be between 0 and 5 (head only)')
    attention_prefix = 'head.refiner.spatial.attention.'
    if coarse_attention_only:
        attention_prefix += 'context_blocks.'
    for name, parameter in model.named_parameters():
        if proposal_size_only:
            enabled = name.startswith('head.refiner.quality.size_context.')
        elif pose_score_only:
            enabled = name.startswith('head.refiner.quality.')
        elif pixel_refinement_only:
            enabled = name.startswith('head.refiner.pixel.')
        elif joint_refinement_only:
            enabled = name.startswith(('head.refiner.local.', 'head.refiner.spatial.'))
        elif spatial_attention_only:
            enabled = name.startswith(attention_prefix)
        elif spatial_rsn_blocks_only:
            enabled = (name.startswith('head.refiner.spatial.rsn_blocks.') or
                       (train_spatial_axis_gain and name == 'head.refiner.spatial.axis_gain'))
        elif simcc_output_only:
            enabled = name.startswith(('head.cls_x.', 'head.cls_y.'))
        elif freeze_pose_confidence and name.startswith('head.refiner.quality.'):
            enabled = False
        elif name.startswith(RSN_PARAMETER_PREFIXES):
            enabled = True
        elif freeze_base:
            enabled = False
        elif name.startswith('backbone.') and train_backbone_from:
            enabled = any(name.startswith(f'backbone.stage{stage}.')
                          for stage in range(train_backbone_from, 5))
        else:
            enabled = True
        parameter.requires_grad_(enabled)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def anchor_trainable_weights(optimizer, initial_groups, strength):
    for group, initial in zip(optimizer.param_groups, initial_groups):
        shrink = strength * group['lr']
        torch._foreach_lerp_(group['params'], initial, shrink/(1 + shrink))


def load_initial_weights(model, state):
    has_branch = any(key.startswith(RSN_PARAMETER_PREFIXES) for key in state)
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(RSN_PARAMETER_PREFIXES) for key in incompatible.missing_keys)
    # BatchNorm can synthesize num_batches_tracked during loading. Inspect the
    # actual source keys rather than comparing missing keys with every buffer.
    assert not incompatible.missing_keys or not has_branch, 'Partial branch checkpoints are not valid initializations'
    return incompatible


def save_best_checkpoint(payload, output, official_ap, already_saved=False):
    with checkpoint_lock(output):
        return _save_best_checkpoint(payload, output, official_ap, already_saved)


def _save_best_checkpoint(payload, output, official_ap, already_saved=False):
    """Publish a complete checkpoint before the global retention collector runs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'best_checkpoint.json'
    ap = float(payload['metrics']['AP'])
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if ap <= previous['native_ap']:
            return False
    path = output/'best_candidate.pth'
    if not already_saved:
        atomic_save(payload, path)
    digest = file_sha256(path)
    history = output/'best_history'
    history.mkdir(exist_ok=True)
    archive = history/f"epoch_{payload['epoch']:03d}_AP{ap*100:.6f}_{digest[:8]}.pth"
    if not archive.exists():
        shutil.copy2(path, archive)
    manifest = dict(epoch=payload['epoch'], native_ap=ap,
                    gain_ap_points=(ap-official_ap)*100,
                    checkpoint=str(path), archive_checkpoint=str(archive), sha256=digest,
                    selection='Largest complete COCO val2017 native AP observed')
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(manifest_path)
    return True


def cosine_tail_lrs(seen_images, schedule):
    progress = (seen_images-schedule['start_images'])/max(
        schedule['end_images']-schedule['start_images'], 1)
    if progress <= 0:
        return list(schedule['start_lrs'])
    if progress >= 1:
        return list(schedule['end_lrs'])
    weight = .5*(1+math.cos(math.pi*progress))
    return [end+(start-end)*weight for start,end in
            zip(schedule['start_lrs'],schedule['end_lrs'])]


def checkpoint_ema_age(checkpoint):
    if 'ema_age_images' in checkpoint:
        return int(checkpoint['ema_age_images'])
    if 'seen_images' in checkpoint:
        return int(checkpoint['seen_images']) + int(checkpoint.get('ema_age_offset', 0))
    if not checkpoint.get('trained_this_phase'):
        return 0
    args = checkpoint.get('args', {})
    return int(checkpoint.get('epoch', 0)) * int(args.get('updates_per_epoch', 0)) * int(args.get('effective_batch', 0))


def planned_training_images(epochs, micro_limit, batch_size, world, effective_batch,
                            schedule_updates=None):
    if schedule_updates is not None:
        if schedule_updates <= 0:
            raise ValueError('Scheduled optimizer updates must be positive')
        return schedule_updates*effective_batch
    return epochs*micro_limit*batch_size*world


def snapshot_source(output):
    destination = output/'source_snapshot'
    paths = list(ROOT.glob('*.py')) + list((ROOT/'tests').glob('*.py'))
    paths += list((ROOT/'configs').glob('*.json')) + [ROOT/'requirements-lock.txt']
    contents = {path.relative_to(ROOT): path.read_bytes() for path in sorted(paths)}
    hashes = {str(path): hashlib.sha256(content).hexdigest() for path, content in contents.items()}
    manifest = destination/'sha256.json'
    if manifest.is_file() and json.loads(manifest.read_text()) != hashes:
        archive = output/'source_snapshot_archive'/file_sha256(manifest)
        if not archive.exists():
            shutil.copytree(destination, archive)
    for relative, content in contents.items():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(hashes, indent=2)+'\n')
    temporary.replace(manifest)
    return manifest


def train_stage(argv):
    parser = argparse.ArgumentParser(description="Run one RSN training stage")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--initialize-from-ema', action='store_true',
                        help='Read the verified EMA weights from a previous phase resume file')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--variant', choices=['rsn', 'rsn_fusion', 'rsn_highres'], default='rsn_fusion')
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--lateral-channels', type=int, default=17)
    parser.add_argument('--spatial-stride', type=int, choices=[8, 16], default=16)
    parser.add_argument('--context-width', type=int, default=0)
    parser.add_argument('--local-width', type=int, default=0)
    parser.add_argument('--trunk-width', type=int, default=0)
    parser.add_argument('--spatial-width', type=int, default=0)
    parser.add_argument('--pixel-width', type=int, default=0)
    parser.add_argument('--rsn-steps', type=int, choices=[2, 3, 4], default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--training-data', choices=['coco_mpii'], default='coco_mpii')
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--backbone-lr', type=float, default=4e-5)
    parser.add_argument('--backbone-layer-decay', type=float, default=1.)
    parser.add_argument('--new-lr', type=float, default=2e-3)
    parser.add_argument('--context-lr', type=float)
    parser.add_argument('--spatial-gain-lr', type=float,
                        help='Learning rate for the learned heatmap-to-SimCC gains')
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--warmup-updates', type=int, default=30)
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--ema-reference-batch', type=int, default=2048)
    parser.add_argument('--inherit-ema-age', action='store_true',
                        help='Continue the averaging schedule of a trained initialization')
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--checkpoint-stages', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--checkpoint-granularity', choices=['stage', 'blocks', 'nested'], default='nested')
    parser.add_argument('--checkpoint-refiner', action='store_true')
    parser.add_argument('--stem-chunk-size', type=int, default=128)
    parser.add_argument('--stage1-chunk-size', type=int, default=128)
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--update-backbone-bn', action='store_true')
    parser.add_argument('--freeze-refiner-bn', action='store_true',
                        help='Keep the RSN BatchNorm running statistics of the initialization')
    parser.add_argument('--initial-weight-anchor', type=float, default=0.,
                        help='L2 regularization toward the initialized trainable weights')
    parser.add_argument('--freeze-base', action='store_true',
                        help='Optimize only the RSN branch, preserving all pretrained RTMPose-X weights')
    parser.add_argument('--train-backbone-from', type=int, choices=range(6), default=0,
                        help='0: all layers; 1..4: unfreeze from that stage; 5: head and refiner only')
    parser.add_argument('--heatmap-loss-weight', type=float, default=0.,
                        help='Optional training-only Gaussian supervision on existing RSN fine maps')
    parser.add_argument('--foreground-loss-weight', type=float, default=0.,
                        help='Penalize heatmap probability outside the original COCO person mask')
    parser.add_argument('--heatmap-sigma', type=float, default=1.,
                        help='Gaussian standard deviation in the RSN heatmap grid')
    parser.add_argument('--heatmap-ohkm-topk', type=int, default=0,
                        help='Train-only RSN hard-joint mining; zero keeps all-joint MSE')
    parser.add_argument('--oks-loss-weight', type=float, default=0.,
                        help='Expected localization quality loss on original keypoint labels')
    parser.add_argument('--pose-oks-localization-weight', type=float, default=0.,
                        help='Supervise the mean expected OKS of each pose using original labels')
    parser.add_argument('--quality-loss-weight', type=float,
                        help='Ground-truth supervision for keypoint confidence in the multiscale model')
    parser.add_argument('--raw-confidence-output', action='store_true',
                        help='Use raw SimCC confidence in validation and deployment')
    parser.add_argument('--pose-oks-confidence', action='store_true',
                        help='Learn a single GT OKS ranking correction per person')
    parser.add_argument('--proposal-confidence', action='store_true',
                        help='Learn pose confidence on shifted and background COCO proposals')
    parser.add_argument('--proposal-size-confidence', action='store_true',
                        help='Use the proposal width and height in the pose confidence head')
    parser.add_argument('--proposal-size-only', action='store_true',
                        help='Freeze the previously learned confidence weights while fitting proposal size')
    parser.add_argument('--object-proposals', type=Path,
                        help='Use original non-person COCO train2017 boxes and emphasize high-scoring negatives')
    parser.add_argument('--pose-score-only', action='store_true',
                        help='Freeze all coordinate weights and BN statistics while training the pose score')
    parser.add_argument('--pixel-refinement-only', action='store_true',
                        help='Train only the local RGB offset branch')
    parser.add_argument('--joint-refinement-only', action='store_true',
                        help='Train the local offsets and spatial heatmaps with fixed BN statistics')
    parser.add_argument('--simcc-output-only', action='store_true',
                        help='Tune the two SimCC coordinate projections with the feature extractor and RSN frozen')
    parser.add_argument('--spatial-attention-width', type=int, default=0,
                        help='Channels in pooled spatial attention on the RSN heatmap features')
    parser.add_argument('--spatial-attention-layers', type=int, default=1,
                        help='Total attention layers, including the final fine-to-coarse layer')
    parser.add_argument('--spatial-rsn-blocks', type=int, default=0,
                        help='Additional three-step RSN blocks operating on the stride-four heatmaps')
    parser.add_argument('--spatial-rsn-width', type=int, default=0, help='Channels per step in the added RSN blocks')
    parser.add_argument('--spatial-rsn-attention-layers', type=int, default=1,
                        help='Attention layers in each added RSN block')
    parser.add_argument('--spatial-rsn-blocks-only', action='store_true',
                        help='Train the added RSN blocks and their attention with fixed BN statistics')
    parser.add_argument('--train-spatial-axis-gain', action='store_true',
                        help='Fit the existing heatmap-to-SimCC gains together with the added RSN blocks')
    parser.add_argument('--spatial-refiner-chunk-size', type=int, default=0,
                        help='Recompute the entire stride-four refiner in chunks with fixed BN')
    parser.add_argument('--spatial-attention-only', action='store_true',
                        help='Optimize only the pooled RSN spatial attention branch')
    parser.add_argument('--coarse-attention-only', action='store_true',
                        help='Optimize only the coarse self-attention blocks during attention warmup')
    parser.add_argument('--local-loss-weight', type=float, default=0.,
                        help='Ground-truth offset loss on local corrections around visible joints')
    parser.add_argument('--pixel-loss-weight', type=float, default=0.,
                        help='Ground-truth offset loss on the local RGB correction')
    parser.add_argument('--augment', choices=['finetune', 'strong', 'clean', 'occlusion', 'pose_quality'],
                        default='finetune')
    parser.add_argument('--bbox-transform', choices=['legacy', 'official'], default='legacy')
    parser.add_argument('--bbox-shift-factor', type=float, default=.16)
    parser.add_argument('--mpii-loss-weight', type=float, default=1.,
                        help='Relative common-joint supervision on MPII; keep positive to use both sources')
    parser.add_argument('--loss-reduction', choices=['official', 'per_sample'], default='official')
    parser.add_argument('--loss-batch-reference', type=int, default=512)
    parser.add_argument('--simcc-sigma-scale', type=float, default=1.)
    parser.add_argument('--simcc-continuous', action='store_true',
                        help='Generate labels at the original sub-bin coordinate')
    parser.add_argument('--area-aware-simcc', action='store_true',
                        help='Adjust training target widths with annotation area and joint OKS scale')
    parser.add_argument('--simcc-hard-joint-weight', type=float, default=0.,
                        help='Fraction of the primary KL loss assigned to top-eight hard joints')
    parser.add_argument('--flip-supervision-weight', type=float, default=0.,
                        help='Fraction of SimCC supervision applied after merging the two training views')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--schedule-updates', type=int,
                        help='Exact optimizer-update horizon for equal-budget data ablations')
    parser.add_argument('--retention-top-k', type=int, default=3)
    parser.add_argument('--skip-val', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--probe-updates', type=int, default=3,
                        help='Test optimizer allocation and steady DDP memory over several updates')
    parser.add_argument('--official-ap', type=float, default=.7881823654889539)
    parser.add_argument('--minimum-gain', type=float, default=.01,
                        help='Strict AP-fraction gain over official; 0.01 means more than 1 AP point')
    parser.add_argument('--continue-after-target', action='store_true',
                        help='Complete the configured schedule while retaining the largest observed AP')
    parser.add_argument('--extend-schedule', action='store_true',
                        help='Extend a resumed cosine schedule smoothly from the current learning rates')
    args = parser.parse_args(argv)
    if args.joint_refinement_only and (not args.local_width or not args.spatial_width or
                                      args.pose_score_only or args.pixel_refinement_only or args.update_backbone_bn):
        parser.error('Joint refinement requires both geometry heads, fixed BN and a single training mode')
    if args.spatial_attention_width and (not args.spatial_width or args.spatial_attention_width < 0
                                        or args.spatial_attention_width % RSNSpatialAttention.heads):
        parser.error('Spatial attention requires a spatial heatmap refiner and a positive width divisible by four')
    if args.spatial_attention_layers < 1 or (args.spatial_attention_layers > 1 and not args.spatial_attention_width):
        parser.error('Additional attention layers require an enabled spatial attention branch')
    if (args.spatial_rsn_blocks < 0 or args.spatial_rsn_width < 0 or args.spatial_rsn_attention_layers < 1 or
            bool(args.spatial_rsn_blocks) != bool(args.spatial_rsn_width) or
            (args.spatial_rsn_blocks and not args.spatial_attention_width)):
        parser.error('Spatial RSN blocks require a positive width and spatial attention')
    if args.spatial_refiner_chunk_size < 0 or (args.spatial_refiner_chunk_size and
            not (args.spatial_width and args.checkpoint_refiner and args.freeze_refiner_bn)):
        parser.error('Spatial refiner chunks require a spatial refiner, checkpointing and fixed BN')
    if args.spatial_attention_only and (not args.spatial_attention_width or args.joint_refinement_only or
                                       args.pose_score_only or args.pixel_refinement_only or args.update_backbone_bn):
        parser.error('Attention-only training requires the attention branch, fixed BN and a single training mode')
    if args.coarse_attention_only and (not args.spatial_attention_only or args.spatial_attention_layers < 2):
        parser.error('Coarse-only warmup requires attention-only training and at least two attention layers')
    if args.simcc_output_only and any((args.joint_refinement_only, args.spatial_attention_only,
                                      args.pose_score_only, args.pixel_refinement_only, args.update_backbone_bn)):
        parser.error('SimCC output tuning requires fixed BN and a single training mode')
    if args.spatial_rsn_blocks_only and (not args.spatial_rsn_blocks or any((
            args.joint_refinement_only, args.spatial_attention_only, args.simcc_output_only,
            args.pose_score_only, args.pixel_refinement_only, args.update_backbone_bn))):
        parser.error('Added-block training requires spatial RSN blocks, fixed BN and a single training mode')
    if args.train_spatial_axis_gain and not args.spatial_rsn_blocks_only:
        parser.error('Training the spatial axis gain requires added-block training')
    refinement_only = (args.joint_refinement_only or args.spatial_attention_only or
                       args.simcc_output_only or args.spatial_rsn_blocks_only)
    training_proposals = args.proposal_confidence and args.pose_score_only
    freeze_pose_confidence = args.proposal_confidence and not args.pose_score_only
    train_quality = bool(args.context_width and not (args.pixel_refinement_only or refinement_only
                                                    or freeze_pose_confidence))
    if args.quality_loss_weight is None:
        args.quality_loss_weight = .1 if train_quality else 0.
    assert args.batch_size > 1 and args.accum_steps > 0
    assert math.isfinite(args.backbone_layer_decay) and 0 < args.backbone_layer_decay <= 1
    assert math.isfinite(args.initial_weight_anchor) and args.initial_weight_anchor >= 0
    assert math.isfinite(args.bbox_shift_factor) and 0 <= args.bbox_shift_factor <= 1
    assert 0 < args.mpii_loss_weight <= 1 and args.minimum_gain >= 0
    assert not (args.update_backbone_bn and (args.stem_chunk_size or args.stage1_chunk_size)), 'Chunking requires frozen backbone BN'
    assert not (args.freeze_base and args.update_backbone_bn)
    assert math.isfinite(args.heatmap_loss_weight) and args.heatmap_loss_weight >= 0
    assert math.isfinite(args.foreground_loss_weight) and args.foreground_loss_weight >= 0
    assert not args.foreground_loss_weight or (args.spatial_width and args.heatmap_loss_weight)
    assert math.isfinite(args.heatmap_sigma) and args.heatmap_sigma > 0
    assert args.heatmap_loss_weight == 0 or args.variant == 'rsn_highres'
    assert 0 <= args.heatmap_ohkm_topk <= 17
    assert args.heatmap_ohkm_topk == 0 or args.heatmap_loss_weight > 0
    assert not args.spatial_width or args.heatmap_ohkm_topk == 0
    assert math.isfinite(args.oks_loss_weight) and args.oks_loss_weight >= 0
    assert math.isfinite(args.pose_oks_localization_weight) and args.pose_oks_localization_weight >= 0
    assert math.isfinite(args.quality_loss_weight) and args.quality_loss_weight >= 0
    assert train_quality == bool(args.quality_loss_weight)
    assert not args.pose_oks_confidence or args.context_width > 0
    assert not args.proposal_confidence or args.pose_oks_confidence
    assert not args.proposal_size_confidence or args.proposal_confidence
    assert not args.proposal_size_only or (args.proposal_size_confidence and args.pose_score_only)
    assert args.object_proposals is None or training_proposals
    assert not (args.area_aware_simcc and args.pose_score_only)
    assert not args.pose_score_only or (args.pose_oks_confidence and not args.update_backbone_bn
                                       and not args.raw_confidence_output
                                       and not (args.heatmap_loss_weight or args.oks_loss_weight or
                                                args.pose_oks_localization_weight or
                                                args.local_loss_weight))
    assert math.isfinite(args.local_loss_weight) and args.local_loss_weight >= 0
    assert not args.local_loss_weight or args.local_width > 0
    assert math.isfinite(args.pixel_loss_weight) and args.pixel_loss_weight >= 0
    assert not args.pixel_loss_weight or args.pixel_width > 0
    assert not args.pixel_refinement_only or (args.pixel_width > 0 and not args.pose_score_only)
    assert math.isfinite(args.simcc_sigma_scale) and args.simcc_sigma_scale > 0
    assert 0 <= args.simcc_hard_joint_weight <= 1
    assert 0 <= args.flip_supervision_weight <= 1
    assert not (args.flip_supervision_weight and args.pose_score_only)
    assert args.context_lr is None or (args.context_width > 0 and math.isfinite(args.context_lr) and args.context_lr > 0)
    assert args.spatial_gain_lr is None or (args.spatial_width > 0 and math.isfinite(args.spatial_gain_lr)
                                           and args.spatial_gain_lr > 0)
    rank, local_rank, world = distributed_setup()
    device = torch.device('cuda', local_rank)
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(args.amp)
    torch.manual_seed(args.seed)
    random.seed(args.seed+rank); np.random.seed(args.seed+rank)
    torch.backends.cudnn.benchmark = args.autotune
    args.output.mkdir(parents=True, exist_ok=True)
    settings = canonical_metadata({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    settings.update(world_size=world, effective_batch=args.batch_size*args.accum_steps*world,
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    training_models=1, datasets=['COCO train2017', 'MPII train'],
                    data_dir=str(DATA),
                    training_views_per_example=2 if args.flip_supervision_weight else 1,
                    selection='Strictly exceed reproduced official COCO val2017 AP; deployment verified separately')
    model = RTMPoseX(args.variant, args.width, args.checkpoint_stages, args.rsn_steps,
                     args.checkpoint_granularity, args.stem_chunk_size, args.stage1_chunk_size,
                     args.lateral_channels, args.spatial_stride, args.context_width, args.local_width,
                     args.trunk_width, args.spatial_width, args.pixel_width, args.spatial_attention_width,
                     args.spatial_attention_layers, args.spatial_rsn_blocks, args.spatial_rsn_width,
                     args.spatial_rsn_attention_layers)
    if args.pose_oks_confidence:
        model.head.enable_pose_oks_confidence(proposal_confidence=args.proposal_confidence,
                                              proposal_size_confidence=args.proposal_size_confidence)
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    initial_args = initial.get('args', {})
    state = initial['ema'] if args.initialize_from_ema else initial.get('model', initial.get('state_dict', initial))
    incompat = load_initial_weights(model, state)
    if initial_args.get('variant', 'baseline') != 'baseline':
        assert not incompat.missing_keys, 'Candidate initialization must include its complete branch'
    initial_metrics = initial.get('metrics', {})
    ema_age_offset = checkpoint_ema_age(initial) if args.inherit_ema_age else 0
    if args.initial_weight_anchor:
        settings['anchor_checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.object_proposals is not None:
        with args.object_proposals.open('rb') as input_file:
            settings['object_proposal_sha256'] = hashlib.file_digest(input_file, 'sha256').hexdigest()
    if incompat.missing_keys and not initial_metrics:
        # The zero output branch exactly preserves the verified official model.
        initial_metrics = json.loads((ROOT/'results/baseline_official.json').read_text())
    best = initial_metrics.get('AP', -1.)
    del initial, state
    settings['trainable_parameters'] = configure_trainable(
        model, args.freeze_base, args.train_backbone_from, args.pose_score_only,
        args.pixel_refinement_only, args.proposal_size_only, freeze_pose_confidence,
        joint_refinement_only=args.joint_refinement_only, spatial_attention_only=args.spatial_attention_only,
        coarse_attention_only=args.coarse_attention_only, simcc_output_only=args.simcc_output_only,
        spatial_rsn_blocks_only=args.spatial_rsn_blocks_only, train_spatial_axis_gain=args.train_spatial_axis_gain)
    model.head.refiner.checkpoint_refinement = args.checkpoint_refiner
    if args.spatial_refiner_chunk_size:
        model.head.refiner.spatial.checkpoint_chunk_size = args.spatial_refiner_chunk_size
    for refiner in model.backbone.stage_refiners.values():
        refiner.checkpoint_refinement = args.checkpoint_refiner
    model.head.raw_confidence_output = args.raw_confidence_output
    model.head.auxiliary_heatmaps = args.heatmap_loss_weight > 0
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model, args)
    initial_groups = None
    if args.initial_weight_anchor:
        initial_groups = [[parameter.detach().clone() for parameter in group['params']]
                          for group in optimizer.param_groups]
    # This copy is a frozen EMA of the one trained network; it has no optimizer.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp == 'fp16', init_scale=1024.)
    updates, seen_images, start_epoch = 0, 0, 0
    previous_schedule, resumed_lrs, previous_epochs = None, None, None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert saved['args'].get('freeze_base', False) == args.freeze_base
        assert saved['args'].get('inherit_ema_age', False) == args.inherit_ema_age
        assert saved['args'].get('freeze_refiner_bn', False) == args.freeze_refiner_bn
        assert saved['args'].get('initial_weight_anchor', 0.) == args.initial_weight_anchor
        if args.initial_weight_anchor:
            assert saved['args']['anchor_checkpoint_sha256'] == settings['anchor_checkpoint_sha256']
        assert saved['args'].get('train_backbone_from', 0) == args.train_backbone_from
        assert saved['args'].get('lateral_channels', 17) == args.lateral_channels
        assert saved['args'].get('spatial_stride', 16) == args.spatial_stride
        assert saved['args'].get('context_width', 0) == args.context_width
        assert saved['args'].get('local_width', 0) == args.local_width
        assert saved['args'].get('trunk_width', 0) == args.trunk_width
        assert saved['args'].get('spatial_width', 0) == args.spatial_width
        assert saved['args'].get('spatial_attention_width', 0) == args.spatial_attention_width
        assert saved['args'].get('spatial_attention_layers', 1) == args.spatial_attention_layers
        assert saved['args'].get('spatial_rsn_blocks', 0) == args.spatial_rsn_blocks
        assert saved['args'].get('spatial_rsn_width', 0) == args.spatial_rsn_width
        assert saved['args'].get('spatial_rsn_attention_layers', 1) == args.spatial_rsn_attention_layers
        assert saved['args'].get('pixel_width', 0) == args.pixel_width
        assert saved['args'].get('context_lr') == args.context_lr
        assert saved['args'].get('spatial_gain_lr') == args.spatial_gain_lr
        assert saved['args'].get('training_data', 'coco_mpii') == args.training_data
        assert saved['args'].get('bbox_transform', 'legacy') == args.bbox_transform
        assert saved['args'].get('bbox_shift_factor', .16) == args.bbox_shift_factor
        assert saved['args'].get('backbone_layer_decay', 1.) == args.backbone_layer_decay
        assert saved['args'].get('checkpoint_refiner', False) == args.checkpoint_refiner
        assert saved['args'].get('schedule_updates') == args.schedule_updates or args.extend_schedule
        assert saved['args'].get('heatmap_loss_weight', 0.) == args.heatmap_loss_weight
        assert saved['args'].get('foreground_loss_weight', 0.) == args.foreground_loss_weight
        assert saved['args'].get('heatmap_sigma', 1.) == args.heatmap_sigma
        assert saved['args'].get('heatmap_ohkm_topk', 0) == args.heatmap_ohkm_topk
        assert saved['args'].get('oks_loss_weight', 0.) == args.oks_loss_weight
        assert saved['args'].get('pose_oks_localization_weight', 0.) == args.pose_oks_localization_weight
        assert saved['args'].get('quality_loss_weight', 0.) == args.quality_loss_weight
        assert saved['args'].get('pose_oks_confidence', False) == args.pose_oks_confidence
        assert saved['args'].get('proposal_confidence', False) == args.proposal_confidence
        assert saved['args'].get('proposal_size_confidence', False) == args.proposal_size_confidence
        assert saved['args'].get('proposal_size_only', False) == args.proposal_size_only
        assert saved['args'].get('object_proposal_sha256') == settings.get('object_proposal_sha256')
        assert saved['args'].get('pose_score_only', False) == args.pose_score_only
        assert saved['args'].get('pixel_refinement_only', False) == args.pixel_refinement_only
        assert saved['args'].get('joint_refinement_only', False) == args.joint_refinement_only
        assert saved['args'].get('spatial_attention_only', False) == args.spatial_attention_only
        assert saved['args'].get('spatial_rsn_blocks_only', False) == args.spatial_rsn_blocks_only
        assert saved['args'].get('train_spatial_axis_gain', False) == args.train_spatial_axis_gain
        assert saved['args'].get('coarse_attention_only', False) == args.coarse_attention_only
        assert saved['args'].get('simcc_output_only', False) == args.simcc_output_only
        assert saved['args'].get('local_loss_weight', 0.) == args.local_loss_weight
        assert saved['args'].get('pixel_loss_weight', 0.) == args.pixel_loss_weight
        assert saved['args'].get('simcc_sigma_scale', 1.) == args.simcc_sigma_scale
        assert saved['args'].get('simcc_continuous', False) == args.simcc_continuous
        assert saved['args'].get('area_aware_simcc', False) == args.area_aware_simcc
        assert saved['args'].get('simcc_hard_joint_weight', 0.) == args.simcc_hard_joint_weight
        assert saved['args'].get('flip_supervision_weight', 0.) == args.flip_supervision_weight
        for key in ['variant', 'width', 'rsn_steps', 'loss_reduction', 'loss_batch_reference', 'effective_batch']:
            assert saved['args'][key] == settings[key], f'Incompatible resume setting: {key}'
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        rates = [group['initial_lr'] for group in optimizer.param_groups]
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        resumed_lrs = [float(group['lr']) for group in optimizer.param_groups]
        for group, rate in zip(optimizer.param_groups, rates):
            group['initial_lr'] = rate
        updates, seen_images = saved['updates'], saved['seen_images']
        ema_age_offset = saved.get('ema_age_offset', 0)
        start_epoch, best = saved['epoch']+1, saved['best_ap']
        previous_schedule = saved.get('lr_schedule')
        previous_epochs = saved['args']['epochs']
        del saved
        manifest = args.output/'best_checkpoint.json'
        if manifest.is_file():
            best = max(best, float(json.loads(manifest.read_text())['native_ap']))
    ema_state = ema_pairs(ema, model)
    classification_loss = partial(simcc_loss, reduction=args.loss_reduction,
                                  sigma_scale=args.simcc_sigma_scale, continuous=args.simcc_continuous,
                                  hard_joint_weight=args.simcc_hard_joint_weight)
    trained = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
    dataset = CocoMpiiPose(args.augment)
    use_annotation_areas = (args.area_aware_simcc or args.foreground_loss_weight > 0 or
                            args.pose_oks_localization_weight > 0 or
                            (args.pose_oks_confidence and args.quality_loss_weight > 0))
    if use_annotation_areas:
        dataset.use_training_annotation_areas(proposals=training_proposals,
                                              object_proposals=args.object_proposals,
                                              foreground=args.foreground_loss_weight > 0)
    dataset.bbox_transform_mode = args.bbox_transform
    dataset.bbox_shift_factor = args.bbox_shift_factor
    dataset.return_proposal_size = args.proposal_size_confidence
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        drop_last=True, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed+rank))
    # Drop the incomplete accumulation window so every actual update has the requested batch.
    micro_limit = len(loader)//args.accum_steps*args.accum_steps
    assert micro_limit > 0
    total_images = planned_training_images(args.epochs, micro_limit, args.batch_size, world,
                                           settings['effective_batch'], args.schedule_updates)
    lr_schedule = None
    if args.resume:
        if previous_schedule and previous_schedule['end_images'] == total_images:
            lr_schedule = previous_schedule
        elif args.extend_schedule:
            assert total_images > seen_images, 'Extended schedule must have remaining training images'
            lr_schedule = dict(kind='anchored_cosine', start_images=seen_images,
                               end_images=total_images, start_lrs=resumed_lrs,
                               end_lrs=[float(group['initial_lr'])*.1 for group in optimizer.param_groups])
        else:
            assert previous_epochs == args.epochs, 'Use --extend-schedule when changing the total epochs'
            assert previous_schedule is None, 'Preserve or explicitly extend the existing anchored schedule'
    settings['lr_schedule'] = lr_schedule
    settings['ema_age_offset'] = ema_age_offset
    settings['source_counts'] = dataset.source_counts
    settings['updates_per_epoch'] = micro_limit//args.accum_steps
    if rank == 0:
        source_manifest = snapshot_source(args.output)
        (args.output/'args.json').write_text(json.dumps(settings, indent=2)+'\n')
        if not args.probe:
            record_session(args.output, settings, source_manifest, start_epoch, updates)
        print('SINGLE_MODEL '+json.dumps(settings), flush=True)
        if not args.resume and not args.probe and initial_metrics:
            atomic_save(dict(model=cpu_state(ema), args=settings, epoch=0, metrics=initial_metrics,
                             trained_this_phase=False), args.output/'best_candidate.pth')
        if args.resume:
            previous = args.resume.parent/'best_candidate.pth'
            current = args.output/'best_candidate.pth'
            if previous.is_file() and not current.exists():
                shutil.copy2(previous, current)
    torch.cuda.reset_peak_memory_stats(device)
    stop = False
    frozen_forward = args.pose_score_only or args.pixel_refinement_only or refinement_only
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch); trained.train()
        if frozen_forward:
            model.eval()
            # Request the training output while keeping the coordinate modules in eval mode.
            model.head.training = True
        elif args.freeze_base:
            model.backbone.eval()
        if not args.update_backbone_bn:
            for name, layer in model.backbone.named_modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm) and not name.startswith('stage_refiners.'):
                    layer.eval()
        if not frozen_forward:
            model.backbone.stage_refiners.train()
        if args.freeze_refiner_bn:
            for refiner in (model.head.refiner, model.backbone.stage_refiners):
                for layer in refiner.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
        optimizer.zero_grad(set_to_none=True)
        losses, supervised_sum, heatmap_sum, oks_sum, quality_sum, local_sum = (0.,) * 6
        pose_oks_sum, pixel_sum = 0., 0.
        source_counts = torch.zeros(2, dtype=torch.long)
        start = time.perf_counter()
        for iteration, batch in enumerate(loader):
            if iteration >= micro_limit:
                break
            images = preprocess(batch['image'], device, args.channels_last)
            metadata = (batch['proposal_size'].to(device, non_blocking=True),) if args.proposal_size_confidence else ()
            if dtype is not None:
                images = images.to(dtype=dtype)
            if args.flip_supervision_weight:
                images = torch.cat((images, images.flip(-1)), dim=0)
                metadata = tuple(torch.cat((value, value), dim=0) for value in metadata)
            keypoints = batch['keypoints'].to(device, non_blocking=True)
            visible = batch['visible'].to(device, non_blocking=True)
            if args.oks_loss_weight or (args.quality_loss_weight and not args.pose_oks_confidence):
                area = batch['person_box_area'].to(device, dtype=torch.float32, non_blocking=True)
            if use_annotation_areas and not training_proposals:
                annotation_area = batch['annotation_area'].to(device, non_blocking=True)
            sources = batch['source'].to(device, non_blocking=True)
            if args.mpii_loss_weight != 1:
                source_weights = torch.where(sources > 0, args.mpii_loss_weight, 1.)
                visible = visible*source_weights[:, None]
            source_counts += batch['source'].bincount(minlength=2)
            if iteration % args.accum_steps == 0:
                if lr_schedule is not None:
                    for group, rate in zip(optimizer.param_groups, cosine_tail_lrs(seen_images, lr_schedule)):
                        group['lr'] = rate
                else:
                    progress = min(seen_images/total_images, 1.)
                    factor = .1 + .9*(1+math.cos(math.pi*progress))*.5
                    warmup_images = args.warmup_updates*settings['effective_batch']
                    if seen_images < warmup_images:
                        factor *= .1 + .9*seen_images/max(warmup_images, 1)
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr']*factor
            sync = (iteration+1) % args.accum_steps == 0
            sync_context = trained.no_sync() if world > 1 and not sync else nullcontext()
            with sync_context:
                with torch.autocast('cuda', dtype=dtype) if dtype else nullcontext():
                    outputs = trained(images, *metadata)
                prediction = outputs['prediction'] if isinstance(outputs, dict) else outputs
                single_prediction = prediction
                if args.flip_supervision_weight:
                    single_prediction, mirrored = zip(*(axis.chunk(2, dim=0) for axis in prediction))
                    prediction = merge_simcc_flip(single_prediction, mirrored)
                loss = 0.
                if not args.pose_score_only:
                    target_area = annotation_area if args.area_aware_simcc else None
                    loss = classification_loss(prediction, keypoints, visible, annotation_area=target_area)
                    if 0 < args.flip_supervision_weight < 1:
                        mirrored_points = keypoints[:, FLIP].clone()
                        mirrored_points[..., 0] = (WIDTH - 1) - mirrored_points[..., 0]
                        direct_loss = classification_loss(single_prediction, keypoints, visible,
                                                          annotation_area=target_area)
                        mirror_loss = classification_loss(mirrored, mirrored_points, visible[:, FLIP],
                                                          annotation_area=target_area)
                        loss = (args.flip_supervision_weight * loss +
                                (1 - args.flip_supervision_weight) * (direct_loss + mirror_loss) * .5)
                    if args.loss_reduction == 'official':
                        loss = loss*(args.loss_batch_reference/args.batch_size)
                    supervised_sum += float(loss.detach())
                if args.heatmap_loss_weight:
                    heatmaps = outputs['auxiliary_heatmaps'][:len(keypoints)]
                    if args.spatial_width:
                        foreground = {}
                        if args.foreground_loss_weight:
                            foreground = {
                                'foreground_mask': batch['foreground_mask'].to(device, non_blocking=True),
                                'foreground_visible': batch['foreground_visible'].to(device, non_blocking=True),
                                'foreground_ratio': args.foreground_loss_weight / args.heatmap_loss_weight,
                            }
                        auxiliary = heatmap_distribution_loss(heatmaps, keypoints, visible,
                                                              model.head.refiner.spatial.logit_scale,
                                                              sigma=args.heatmap_sigma, **foreground)
                    else:
                        auxiliary = heatmap_loss(heatmaps, keypoints, visible,
                                                 args.loss_batch_reference, sigma=args.heatmap_sigma,
                                                 ohkm_topk=args.heatmap_ohkm_topk)
                    heatmap_sum += float(auxiliary.detach())
                    loss = loss + args.heatmap_loss_weight*auxiliary
                if args.oks_loss_weight:
                    localization = simcc_oks_loss(prediction, keypoints, visible, area)
                    oks_sum += float(localization.detach())
                    loss = loss + args.oks_loss_weight*localization
                if args.pose_oks_localization_weight:
                    pose_oks = simcc_pose_oks_loss(prediction, keypoints, visible, annotation_area)
                    pose_oks_sum += float(pose_oks.detach())
                    loss = loss + args.pose_oks_localization_weight*pose_oks
                if args.quality_loss_weight:
                    score_delta = outputs['score_delta'][:len(keypoints)]
                    if args.proposal_confidence:
                        proposal_points = batch['proposal_keypoints'].to(device, non_blocking=True)
                        proposal_areas = batch['proposal_areas'].to(device, non_blocking=True)
                        proposal_weights = batch['proposal_weight'].to(device, dtype=torch.float32, non_blocking=True)
                        proposal_weights *= torch.where(sources > 0, args.mpii_loss_weight, 1.)
                        quality = pose_proposal_loss(single_prediction, score_delta, proposal_points,
                                                     proposal_areas, proposal_weights,
                                                     hard_negatives=args.object_proposals is not None)
                    elif args.pose_oks_confidence:
                        quality = pose_oks_rank_loss(single_prediction, score_delta, keypoints,
                                                     visible, annotation_area, sources)
                    else:
                        quality = pose_quality_loss(single_prediction, score_delta, keypoints, visible, area)
                    quality_sum += float(quality.detach())
                    loss = loss + args.quality_loss_weight*quality
                if args.local_loss_weight:
                    anchor, offset = (value[:len(keypoints)] for value in outputs['local_refinement'])
                    local = local_joint_loss(anchor, offset, keypoints, visible, model.head.refiner.local.max_shift)
                    local_sum += float(local.detach())
                    loss = loss + args.local_loss_weight*local
                if args.pixel_loss_weight:
                    anchor, offset = (value[:len(keypoints)] for value in outputs['pixel_refinement'])
                    pixel = local_joint_loss(anchor, offset, keypoints, visible, model.head.refiner.pixel.max_shift)
                    pixel_sum += float(pixel.detach())
                    loss = loss + args.pixel_loss_weight*pixel
                losses += float(loss.detach())
                scaler.scale(loss/args.accum_steps).backward()
            if not sync:
                continue
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 35.)
            finite = bool(torch.isfinite(norm))
            if not finite and args.amp != 'fp16':
                raise FloatingPointError('Non-finite candidate gradient')
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            seen_images += settings['effective_batch']
            if finite:
                if initial_groups is not None:
                    anchor_trainable_weights(optimizer, initial_groups, args.initial_weight_anchor)
                updates += 1
                virtual = (seen_images + ema_age_offset)/args.ema_reference_batch
                decay = min(args.ema_decay, (1+virtual)/(10+virtual)) ** (settings['effective_batch']/args.ema_reference_batch)
                update_ema(ema_state, decay)
            if (args.probe and updates >= args.probe_updates) or (args.max_updates and updates >= args.max_updates):
                stop = True
            if stop or updates <= 2 or updates % 20 == 0 or iteration+1 == micro_limit or args.probe:
                elapsed = time.perf_counter()-start
                row = dict(epoch=epoch+1, update=updates, seen_images=seen_images,
                           microbatch=iteration+1, loss=losses/(iteration+1),
                           supervised_loss=supervised_sum/(iteration+1),
                           auxiliary_heatmap_loss=heatmap_sum/(iteration+1),
                           oks_localization_loss=oks_sum/(iteration+1),
                           pose_oks_localization_loss=pose_oks_sum/(iteration+1),
                           keypoint_quality_loss=quality_sum/(iteration+1),
                           local_refinement_loss=local_sum/(iteration+1),
                           pixel_refinement_loss=pixel_sum/(iteration+1),
                           gradient_norm=float(norm), optimizer_step_skipped=not finite,
                           learning_rates=[float(group['lr']) for group in optimizer.param_groups],
                           images_per_second=(iteration+1)*args.batch_size*world/elapsed,
                           seconds=elapsed, allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                           reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30)
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    with (args.output/'train.jsonl').open('a') as output:
                        output.write(json.dumps(row)+'\n')
            if stop:
                break
        counts = source_counts.to(device)
        if world > 1:
            dist.all_reduce(counts)
        if rank == 0:
            print('DATA_SEEN '+json.dumps(dict(epoch=epoch+1, COCO=int(counts[0]), MPII=int(counts[1]))), flush=True)
        if not args.skip_val and not args.probe:
            metrics = run_evaluation(ema, args.output/f'epoch_{epoch+1:03d}_candidate',
                                     batch_size=32, workers=args.workers, flip=True, amp='fp16')
            packet = [metrics]
            if world > 1:
                dist.broadcast_object_list(packet, src=0)
            metrics = packet[0]
            if metrics['AP'] > best:
                best = metrics['AP']
                if rank == 0:
                    save_best_checkpoint(dict(model=cpu_state(ema), args=settings, epoch=epoch+1,
                                         metrics=metrics, trained_this_phase=True,
                                         ema_age_images=seen_images+ema_age_offset), args.output, args.official_ap)
            accuracy_passed = best > args.official_ap+args.minimum_gain
            if rank == 0:
                event = dict(epoch=epoch+1, updates=updates, metrics=metrics, best_ap=best,
                             accuracy_passed=accuracy_passed, official_ap=args.official_ap,
                             minimum_ap_gain=args.minimum_gain)
                with (args.output/'evaluations.jsonl').open('a') as output:
                    output.write(json.dumps(event)+'\n')
                print('EVALUATION '+json.dumps(event), flush=True)
                if accuracy_passed:
                    (args.output/'CANDIDATE_PASSED.json').write_text(json.dumps(event, indent=2))
            if accuracy_passed and not args.continue_after_target:
                stop = True
        if rank == 0 and not args.probe:
            atomic_save(dict(model=cpu_state(model), ema=cpu_state(ema),
                             optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                             epoch=epoch, updates=updates, seen_images=seen_images,
                             ema_age_offset=ema_age_offset,
                             best_ap=best, args=settings, lr_schedule=lr_schedule), args.output/'last.pth')
            if not args.skip_val:
                archive_ranked_epoch(args.output, event, args.retention_top_k)
        if world > 1:
            dist.barrier()
        if stop:
            break
    if rank == 0:
        print('SINGLE_TRAINING_FINISHED '+json.dumps(dict(best_ap=best, updates=updates, probe=args.probe)), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def plan_stages(recipe, gpus, batch_size=None):
    architecture = recipe['architecture'].copy()
    plans = []
    names = set()
    if gpus < 1 or (batch_size is not None and batch_size < 2):
        raise ValueError('GPU count and batch sizes must be positive, with at least two samples per GPU')
    for stage in recipe['stages']:
        effective_batch = stage['effective_batch']
        name = stage['name']
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in names:
            raise ValueError(f'Invalid or duplicate stage name: {name}')
        names.add(name)
        previous = architecture.copy()
        architecture.update(stage.get('architecture', {}))
        options = {**recipe['common'], **stage['options'], **architecture}
        physical = options['batch_size'] if batch_size is None else batch_size
        if physical < 2:
            raise ValueError(f'{name}: the per-GPU batch must contain at least two samples')
        accumulation, remainder = divmod(effective_batch, physical * gpus)
        if accumulation < 1 or remainder:
            raise ValueError(f'{name}: per-GPU batch × GPUs must divide effective batch {effective_batch}')
        if not 0 < stage['handoff_epoch'] <= options['epochs'] or stage['handoff_updates'] <= 0:
            raise ValueError(f'{name}: invalid handoff in the planned schedule')
        options.update(batch_size=physical, accum_steps=accumulation,
                       max_updates=stage['handoff_updates'], continue_after_target=True)
        if options.get('training_data') != 'coco_mpii':
            raise ValueError('The release pipeline requires original COCO and MPII supervision')
        initialization = 'official' if not plans else ('migrate' if architecture != previous else 'reuse')
        plans.append(dict(stage, initialization=initialization, architecture=architecture.copy(), options=options))
    if not plans:
        raise ValueError('The training recipe has no stages')
    return plans


def stage_command(stage, checkpoint, directory, data, gpus, resume=False):
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               f'--nproc-per-node={gpus}', str(Path(__file__).resolve()), '_stage', '--checkpoint', str(checkpoint),
               '--output', str(directory)]
    if resume:
        command += ['--resume', str(directory/'last.pth')]
    for name, value in stage['options'].items():
        if value is None or value is False:
            continue
        command.append('--'+name.replace('_', '-'))
        if value is True:
            continue
        if name == 'object_proposals':
            value = data/value
        command.extend(map(str, value if isinstance(value, list) else [value]))
    return command


def initialize_stage(source, checkpoint, stage, source_hash):
    arch = stage['architecture']
    migration = {key: value for key, value in arch.items()
                 if key not in ('variant', 'rsn_steps', 'raw_confidence_output')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stage['options']['seed'])
        if stage['initialization'] == 'official':
            confidence_keys = {'pose_oks_confidence', 'proposal_confidence', 'proposal_size_confidence'}
            kwargs = {key: value for key, value in migration.items() if key not in confidence_keys}
            model = RTMPoseX(arch['variant'], num_steps=arch['rsn_steps'], **kwargs)
            model.load_official(source)
        else:
            model = capacity_candidate(load_model(source), **migration)
    model.head.raw_confidence_output = arch.get('raw_confidence_output', False)
    atomic_save(dict(model=model.state_dict(), args=arch, metrics={}, trained_this_phase=False,
                     initialization_source=str(source), initialization_source_sha256=source_hash), checkpoint)


def run_trainer(command, log, environment):
    with log.open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(command)+'\n')
        stream.flush()
        trainer = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)

        def interrupt(signum, _frame):
            try:
                if trainer.poll() is None:
                    os.killpg(trainer.pid, signal.SIGTERM)
                trainer.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(trainer.pid, signal.SIGKILL)
                trainer.wait()
            except ProcessLookupError:
                trainer.wait()
            raise SystemExit(128 + signum)

        previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = trainer.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if code:
            raise RuntimeError(f'Trainer exited with code {code}. See {log}')


def completed_handoff(run, stage):
    saved = torch.load(run/'last.pth', map_location='cpu', weights_only=True, mmap=True)
    epoch, updates = saved['epoch'] + 1, saved['updates']
    if (epoch, updates) != (stage['handoff_epoch'], stage['handoff_updates']):
        raise ValueError(f'{run}: optimizer checkpoint is at epoch {epoch}, update {updates}')
    validation = next((row for row in reversed(records(run/'evaluations.jsonl'))
                       if (row['epoch'], row['updates']) == (epoch, updates)
                       and row['metrics'].get('flip') and model_score(row) is not None), None)
    if validation is None:
        raise ValueError(f'{run}: the handoff must have complete native COCO validation')
    selected = run/'handoff.pth'
    atomic_save(dict(model=saved['ema'], args=saved['args'], metrics=validation['metrics'], epoch=epoch,
                     updates=updates, trained_this_phase=True,
                     ema_age_images=saved['seen_images']+saved.get('ema_age_offset', 0)), selected)
    return dict(stage=stage['name'], epoch=epoch, updates=updates, metrics=validation['metrics'],
                checkpoint=str(selected.relative_to(run.parent)), sha256=file_sha256(selected))


def run_pipeline(plans, base, data, output, gpus, output_was_empty):
    files = ['annotations/person_keypoints_train2017.json', 'annotations/person_keypoints_val2017.json',
             'image_index.json', 'mpii_image_index.json', 'train_records.npz', 'mpii_train_records.npz',
             'val_records.npz']
    files += sorted({plan['options']['object_proposals'] for plan in plans if plan['options'].get('object_proposals')})
    counts = []
    for name in ('train_records.npz', 'mpii_train_records.npz'):
        with np.load(data/name) as arrays:
            counts.append(len(arrays['image_ids']))
    for stage in plans:
        updates_per_epoch = sum(counts) // stage['effective_batch']
        if stage['handoff_updates'] != stage['handoff_epoch'] * updates_per_epoch:
            raise ValueError(f"{stage['name']}: dataset counts do not match the recorded epoch length")
    specification = dict(plans=plans, gpus=gpus, base_sha256=file_sha256(base),
                         data={name:file_sha256(data/name) for name in files},
                         sources={path.name:file_sha256(path) for path in sorted(ROOT.glob('*.py'))})
    state_path = output/'pipeline.json'
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state['specification'] != specification:
            raise ValueError('Recipe, runtime batch or input fingerprints changed. Use a new output directory.')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if not output_was_empty:
            raise ValueError(f'{output}: expected a new directory or an existing pipeline.json')
        state = dict(specification=specification, completed=[])
        write_json(state_path, state)

    environment = os.environ.copy()
    environment['RSN_RTMPOSE_DATA'] = str(data)
    source, source_hash = base, specification['base_sha256']
    for index, stage in enumerate(plans):
        if index < len(state['completed']):
            event = state['completed'][index]
            source, source_hash = output/event['checkpoint'], event['sha256']
            if event['stage'] != stage['name'] or file_sha256(source) != source_hash:
                raise ValueError(f'Completed stage checkpoint is inconsistent: {source}')
            continue
        directory = output/stage['name']
        directory.mkdir(exist_ok=True)
        checkpoint = source
        if stage['initialization'] != 'reuse':
            checkpoint = directory/'expanded_initialization.pth'
            if not checkpoint.is_file():
                if (directory/'last.pth').exists():
                    raise FileNotFoundError(f'The resumed stage requires {checkpoint}')
                initialize_stage(source, checkpoint, stage, source_hash)
        checkpoint_hash = file_sha256(checkpoint)
        active = state.get('active')
        if active and (active['stage'] != stage['name'] or active['checkpoint_sha256'] != checkpoint_hash
                       or active['source_sha256'] != source_hash):
            raise ValueError(f'{directory}: recorded initialization changed since the previous launch')
        resume = directory/'last.pth'
        updates = 0
        if resume.is_file():
            updates = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)['updates']
        if updates < stage['handoff_updates']:
            command = stage_command(stage, checkpoint, directory, data, gpus, resume.is_file())
            state['active'] = dict(stage=stage['name'], checkpoint_sha256=checkpoint_hash,
                                   source_sha256=source_hash, command=command,
                                   started_utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
            print(f"Training {stage['name']} at update {updates}. Log: {directory/'train.log'}", flush=True)
            run_trainer(command, directory/'train.log', environment)
        result = completed_handoff(directory, stage)
        result.update(initialization_sha256=checkpoint_hash, source_sha256=source_hash,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        source, source_hash = output/result['checkpoint'], result['sha256']
        state['completed'].append(result)
        state.pop('active', None)
        write_json(state_path, state)
        resume.unlink(missing_ok=True)
        if stage['initialization'] != 'reuse':
            checkpoint.unlink(missing_ok=True)
        print(json.dumps(result), flush=True)
    print(f'Final handoff: {source}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Original RTMPose-X Body7 checkpoint or its tensor-only conversion')
    parser.add_argument('--output', type=Path, required=True, help='Run directory, reused to resume')
    parser.add_argument('--data-dir', type=Path, default=DATA, help='Prepared cache or preparation destination')
    parser.add_argument('--coco-archive', type=Path)
    parser.add_argument('--coco-images', type=Path)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--mpii-annotations', type=Path)
    parser.add_argument('--mpii-images', type=Path)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--batch-size', type=int, help='Override per-GPU crop batch and recalculate accumulation')
    parser.add_argument('--dry-run', action='store_true', help='Print every stage command without writing files')
    args = parser.parse_args()
    source, data, output = (path.expanduser().resolve() for path in (args.checkpoint, args.data_dir, args.output))
    plans = plan_stages(RELEASE_RECIPE, args.gpus, args.batch_size)
    if args.dry_run:
        for stage in plans:
            directory = output/stage['name']
            checkpoint = source if stage['initialization'] == 'reuse' else directory/'expanded_initialization.pth'
            print(json.dumps(dict(stage=stage['name'], initialization=stage['initialization'], source=str(source),
                                  effective_batch=stage['effective_batch'], handoff_epoch=stage['handoff_epoch'],
                                  handoff_updates=stage['handoff_updates'],
                                  command=stage_command(stage, checkpoint, directory, data, args.gpus))))
            source = directory/'handoff.pth'
        return

    if torch.cuda.device_count() < args.gpus:
        parser.error(f'{args.gpus} CUDA GPUs requested, {torch.cuda.device_count()} visible')
    inputs = (args.coco_archive, args.coco_images, args.detections, args.mpii_annotations, args.mpii_images)
    if any(inputs) and not all(inputs):
        parser.error('Provide all five raw dataset inputs together')
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f'.{output.name}.train.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = (data/'train_records.npz').is_file() and (data/'mpii_train_records.npz').is_file()
        if not prepared:
            if not all(inputs):
                parser.error('Provide a prepared COCO + MPII cache or the five raw dataset inputs')
            prepare_data(argparse.Namespace(output=data, coco_archive=args.coco_archive,
                                           coco_images=args.coco_images, detections=args.detections,
                                           mpii_annotations=args.mpii_annotations, mpii_images=args.mpii_images))
        empty = not output.exists() or not any(output.iterdir())
        base = source
        if file_sha256(source) == OFFICIAL_SHA256:
            base = output/'rtmpose-x-state-only.pth'
            if not base.is_file():
                convert_official(source, base)
            empty = empty or all(entry == base for entry in output.iterdir())
        run_pipeline(plans, base, data, output, args.gpus, empty)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_stage':
        train_stage(sys.argv[2:])
    else:
        main()
