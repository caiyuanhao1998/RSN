# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""COCO 2017 keypoints, reusing corresponding COCO 2014 image files.

Crop, normalization and SimCC conventions follow the pinned MMPose reference.
Training augmentations are explicit shared experiment settings, not hidden in
framework defaults. Validation data is never added to the training records.
"""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('RSN_RTMPOSE_DATA', ROOT/'data')).expanduser().resolve()
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                   1.07, 1.07, .87, .87, .89, .89], dtype=np.float32) / 10
WIDTH, HEIGHT = 288, 384
MAX_PROPOSAL_PEOPLE = 8


def annotated_people(annotations):
    return [row for row in annotations if not row.get('iscrowd', 0)
            and row['num_keypoints'] > 0 and row['bbox'][2] > 0 and row['bbox'][3] > 0]


def prepare_records(image_root='/mnt/realccvl15/ycai51/RSN-data/COCO/images', target=DATA):
    target = Path(target)
    image_root = Path(image_root).resolve()
    images = {}
    for split in ('train2014', 'val2014', 'train2017', 'val2017'):
        for p in (image_root / split).glob('*.jpg'):
            images[int(p.stem.rsplit('_', 1)[1])] = str(p)
    (target / 'image_index.json').write_text(json.dumps(images))
    splits = {}
    for split in ['train', 'val']:
        annotation = json.loads((target / f'annotations/person_keypoints_{split}2017.json').read_text())
        ids = {x['id'] for x in annotation['images']}
        assert not ids.difference(images), f'Missing {len(ids.difference(images))} {split} images'
        splits[split] = ids
        if split == 'train':
            records = annotated_people(annotation['annotations'])
            np.savez(target / 'train_records.npz',
                     image_ids=np.array([a['image_id'] for a in records], np.int64),
                     boxes=np.array([a['bbox'] for a in records], np.float32),
                     keypoints=np.array([a['keypoints'] for a in records], np.float32).reshape(-1, 17, 3))
    assert not splits['train'].intersection(splits['val']), 'Train/validation leakage'
    detections = json.loads((target / 'COCO_val2017_detections_AP_H_56_person.json').read_text())
    det = [d for d in detections if d['category_id'] == 1 and d['image_id'] in splits['val']]
    assert all(d['bbox'][2] > 0 and d['bbox'][3] > 0 for d in det)
    np.savez(target / 'val_records.npz', image_ids=np.array([d['image_id'] for d in det], np.int64),
             boxes=np.array([d['bbox'] for d in det], np.float32),
             scores=np.array([d['score'] for d in det], np.float32))
    report = dict(train_images=len(splits['train']), val_images=len(splits['val']),
                  train_people=len(records), val_detections=len(det), overlap=0,
                  source_images='COCO 2017 image IDs with support for matching 2014 files')
    (target / 'data_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def fix_aspect(scale):
    w, h = scale
    return np.array([max(w, h * .75), max(h, w / .75)], dtype=np.float32)


def affine_matrix(center, scale, rotation=0):
    # Same three-point affine convention as MMPose get_warp_matrix.
    angle = np.deg2rad(rotation)
    src_dir = np.array([math.sin(angle) * scale[0] * .5,
                        -math.cos(angle) * scale[0] * .5], dtype=np.float32)
    src = np.array([center, center + src_dir, center + src_dir + [-src_dir[1], src_dir[0]]], np.float32)
    dst_center = np.array([WIDTH * .5, HEIGHT * .5], np.float32)
    dst_dir = np.array([0, -WIDTH * .5], np.float32)
    dst = np.array([dst_center, dst_center + dst_dir,
                    dst_center + dst_dir + [-dst_dir[1], dst_dir[0]]], np.float32)
    return cv2.getAffineTransform(src, dst)


def official_bbox_transform(center, scale, shift_factor=.16):
    """Pinned MMPose RandomBBoxTransform: truncated normal, shift/scale/rotate.

    RTMPose-X uses scale [.5,1.5], rotation factor 90, shift probability .3,
    rotation probability .6, and disables shift in its final training stage.
    """
    from scipy.stats import truncnorm
    random_v = truncnorm.rvs(-1., 1., size=4).astype(np.float32)
    offset = random_v[:2]*shift_factor if np.random.rand() < .3 else np.zeros(2, np.float32)
    multiplier = random_v[2]*.5+1.
    np.random.rand()  # Official scale_prob=1 still draws this probability.
    rotation = float(random_v[3]*90) if np.random.rand() < .6 else 0.
    return center+offset*scale, scale*multiplier, rotation


class CocoPose(Dataset):
    def __init__(self, split='train', augment='finetune', subset=None):
        self.split, self.augment = split, augment
        self.bbox_transform_mode = 'legacy'
        self.bbox_shift_factor = .16
        self.return_geometry = False
        self.return_proposal_size = False
        self.image_index = json.loads((DATA/'image_index.json').read_text())
        arrays = np.load(DATA/f'{split}_records.npz')
        self.ids = arrays['image_ids']
        self.boxes = arrays['boxes']
        self.keypoints = arrays['keypoints'] if split == 'train' else None
        self.scores = arrays['scores'] if split == 'val' else None
        self.annotation_areas = None
        self.proposal_people = None
        self.object_proposals = None
        self.foreground_polygons = None
        self.last_val_image = (None, None)
        self.indices = np.arange(len(self.ids)) if subset is None else np.array(subset)
        cv2.setNumThreads(0)

    def use_training_annotation_areas(self, proposals=False, object_proposals=None, foreground=False):
        if self.split != 'train':
            raise ValueError('Pose score supervision uses training annotations only')
        annotation = json.loads((DATA/'annotations/person_keypoints_train2017.json').read_text())
        people = annotated_people(annotation['annotations'])
        coco = self.ids >= 0
        ids = np.array([row['image_id'] for row in people], dtype=np.int64)
        boxes = np.array([row['bbox'] for row in people], dtype=np.float32)
        joints = np.array([row['keypoints'] for row in people], dtype=np.float32).reshape(-1, 17, 3)
        if not (np.array_equal(ids, self.ids[coco]) and np.array_equal(boxes, self.boxes[coco]) and
                np.array_equal(joints, self.keypoints[coco])):
            raise ValueError('COCO training records are incompatible with their source annotations')
        # MPII supplies person boxes but has no segmentation area.
        areas = self.boxes[:, 2:].prod(axis=1)
        areas[coco] = np.array([row['area'] for row in people], dtype=np.float32)
        if not (np.isfinite(areas).all() and (areas > 0).all()):
            raise ValueError('Invalid training person area')
        self.annotation_areas = areas
        if foreground:
            self.foreground_polygons = {
                index: tuple(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                             for polygon in row['segmentation'] if len(polygon) >= 6)
                for index, row in zip(np.flatnonzero(coco), people)
            }
        if proposals:
            self.proposal_people = defaultdict(list)
            for index in np.flatnonzero(coco):
                self.proposal_people[int(self.ids[index])].append(index)
            ignored_boxes = defaultdict(list)
            for row in annotation['annotations']:
                if row.get('iscrowd', 0) or row['num_keypoints'] == 0:
                    box = row['bbox']
                    if box[2] > 0 and box[3] > 0:
                        ignored_boxes[row['image_id']].append(box)
            self.proposal_ignored = {image: np.asarray(boxes, dtype=np.float32)
                                     for image, boxes in ignored_boxes.items()}
        if object_proposals is not None:
            with np.load(object_proposals, allow_pickle=False) as objects:
                object_ids = objects['image_ids']
                boxes = objects['boxes']
                categories = objects['category_ids']
            train_ids = {row['id'] for row in annotation['images']}
            if not (proposals and len(object_ids) and object_ids.ndim == 1 and
                    boxes.shape == (len(object_ids), 4) and categories.shape == object_ids.shape and
                    np.isfinite(boxes).all() and (boxes[:, 2:] > 0.).all() and (categories != 1).all() and
                    set(object_ids).issubset(train_ids)):
                raise ValueError('Expected labeled non-person object boxes from COCO train2017')
            self.object_proposals = object_ids, boxes

    def proposal_labels(self, index, image_id, center, scale, flip_width, warp):
        """Collect annotated people in the proposal and mask ambiguous crowd crops."""
        crop_center = center.copy()
        if flip_width:
            crop_center[0] = flip_width - 1 - crop_center[0]
        crop_origin = crop_center - scale * .5

        def intersection(boxes):
            extent = (np.minimum(boxes[:, :2] + boxes[:, 2:], crop_origin + scale) -
                      np.maximum(boxes[:, :2], crop_origin))
            return np.clip(extent, 0, None).prod(1)

        keep = True
        if image_id >= 0:
            people = np.asarray(self.proposal_people.get(image_id, []), dtype=np.int64)
            people = people[intersection(self.boxes[people]) > 0]
            ignored = self.proposal_ignored.get(image_id)
            if ignored is not None:
                overlap = intersection(ignored)
                keep = not np.any((overlap > .1 * scale.prod()) |
                                  (overlap > .3 * ignored[:, 2:].prod(1)))
        else:
            people = np.array([index])
        joints = np.zeros((MAX_PROPOSAL_PEOPLE, 17, 3), dtype=np.float32)
        areas = np.ones(MAX_PROPOSAL_PEOPLE, dtype=np.float32)
        if len(people) > MAX_PROPOSAL_PEOPLE:
            keep = False
        elif len(people):
            selected = self.keypoints[people].copy()
            if flip_width:
                selected[..., 0] = flip_width - 1 - selected[..., 0]
                selected = selected[:, FLIP]
            selected[..., :2] = cv2.transform(selected[..., :2].reshape(-1, 1, 2), warp).reshape(-1, 17, 2)
            joints[:len(people)] = selected
            areas[:len(people)] = self.annotation_areas[people] * (WIDTH / float(scale[0]))**2
        return dict(proposal_keypoints=joints, proposal_areas=areas, proposal_weight=float(keep))

    def foreground_labels(self, index, warp, flip_width, joints, visibility):
        """Rasterize the annotated person's silhouette on the stride-four grid."""
        height, width = HEIGHT // 4, WIDTH // 4
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = []
        for polygon in self.foreground_polygons.get(index, ()):
            if flip_width:
                polygon = polygon.copy()
                polygon[:, 0] = flip_width - 1 - polygon[:, 0]
            positions = cv2.transform(polygon[None], warp)[0] / 4.
            polygons.append(np.rint(positions).astype(np.int32))
        if polygons:
            # Rasterize each component as a union, including overlapping components.
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], 1)
            mask = cv2.dilate(mask, None)
        inside = ((joints >= 0) & (joints < [WIDTH, HEIGHT])).all(-1)
        cells = np.clip(np.rint(np.nan_to_num(joints) / 4.), 0, [width - 1, height - 1]).astype(np.intp)
        supervised = (visibility == 2) & inside & (mask[cells[:, 1], cells[:, 0]] > 0)
        return dict(foreground_mask=mask, foreground_visible=supervised)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = int(self.indices[index])
        image_id = int(self.ids[index])
        box = self.boxes[index]
        object_crop = self.object_proposals is not None and image_id >= 0 and np.random.rand() < .25
        if object_crop:
            images, boxes = self.object_proposals
            object_index = np.random.randint(len(images))
            image_id, box = int(images[object_index]), boxes[object_index]
        path = self.image_index[str(image_id)]
        previous_id, previous_image = self.last_val_image
        if self.split == 'val' and image_id == previous_id:
            image = previous_image
        else:
            image = cv2.imread(path)
            if image is None:
                raise OSError(path)
            if self.split == 'val':
                self.last_val_image = (image_id, image)
        center, scale = box[:2] + box[2:] * .5, box[2:] * 1.25
        center, scale = center.copy(), scale.copy()
        rotation = 0.
        flipped = False
        if self.split == 'train':
            proposal = self.proposal_people is not None and image_id >= 0
            if proposal and not object_crop:
                kind = np.random.rand()
                if kind < .45:
                    center += np.random.uniform(-.06, .06, size=2) * scale
                    scale *= np.random.uniform(.9, 1.1)
                elif kind < .8:
                    center += np.random.uniform(-.9, .9, size=2) * scale
                    scale *= np.random.uniform(.55, 1.15)
                else:
                    center[:] = np.random.uniform(0, 1, size=2) * np.asarray(image.shape[1::-1])
                    scale *= np.random.uniform(.45, 1.15)
                np.clip(center, 0, np.asarray(image.shape[1::-1]) - 1, out=center)
            target = np.zeros((17, 3), dtype=np.float32) if object_crop else self.keypoints[index]
            joints = target[:, :2].copy()
            visible = (target[:, 2] > 0).astype(np.float32)
            if np.random.rand() < .5:
                flipped = True
                image = image[:, ::-1]
                center[0] = image.shape[1] - 1 - center[0]
                joints[:, 0] = image.shape[1] - 1 - joints[:, 0]
                joints, visible = joints[FLIP], visible[FLIP]
            if not proposal and self.augment != 'pose_quality' and visible.sum() >= 9 and np.random.rand() < .3:
                upper = [i for i in range(11) if visible[i]]
                lower = [i for i in range(11, 17) if visible[i]]
                selected = upper if len(upper) >= 2 and (len(lower) < 3 or np.random.rand() < .7) else lower
                if len(selected) >= 2:
                    subset = joints[selected]
                    center = subset.mean(0)
                    scale = (subset.max(0) - subset.min(0)) * 1.5
            if not proposal:
                if self.augment == 'pose_quality':
                    if np.random.rand() < .5:
                        jitter = np.clip(np.random.normal(0., .5, size=2), -1., 1.).astype(np.float32)
                        center += jitter * self.bbox_shift_factor * scale
                        scale *= np.random.uniform(.8, 1.25)
                elif self.bbox_transform_mode == 'official':
                    center, scale, rotation = official_bbox_transform(center, scale, self.bbox_shift_factor)
                else:
                    strong = self.augment == 'strong'
                    scale *= np.clip(1 + np.random.randn() * (.3 if strong else .15),
                                     .5 if strong else .75, 1.5 if strong else 1.25)
                    if np.random.rand() < .6:
                        rotation = np.clip(np.random.randn() * (45 if strong else 20),
                                           -90 if strong else -40, 90 if strong else 40)
        proposal_size = scale / 1.25 if self.return_proposal_size else None
        scale = fix_aspect(scale)
        warp = affine_matrix(center, scale, rotation)
        cropped = cv2.warpAffine(image, warp, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        if self.split == 'train':
            joints = cv2.transform(joints[None], warp)[0]
            if self.augment != 'pose_quality' and np.random.rand() < .5:
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-5, 5)) % 180
                hsv[:, :, 1] *= np.random.uniform(.7, 1.3)
                hsv[:, :, 2] *= np.random.uniform(.7, 1.3)
                cropped = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            if self.augment not in {'clean', 'pose_quality'}:
                occlusion = self.augment == 'occlusion'
                if np.random.rand() < .1:
                    cropped = cv2.GaussianBlur(cropped, (3, 3), 0)
                if occlusion and np.random.rand() < .1:
                    cropped = cv2.medianBlur(cropped, 3)
                if np.random.rand() < (.60 if occlusion else .30):
                    limits = (.20, .40) if occlusion else (.10, .30)
                    h = int(HEIGHT * np.random.uniform(*limits))
                    w = int(WIDTH * np.random.uniform(*limits))
                    y, x = np.random.randint(HEIGHT - h), np.random.randint(WIDTH - w)
                    cropped[y:y+h, x:x+w] = 0
        sample = dict(image=torch.from_numpy(cropped.transpose(2, 0, 1).copy()),
                      center=center.astype(np.float32), scale=scale.astype(np.float32),
                      image_id=image_id, index=index)
        if proposal_size is not None:
            sample['proposal_size'] = np.log2(np.maximum(proposal_size, 1.) / 64.).astype(np.float32)
        if self.split == 'train':
            crop_area_scale = (WIDTH / max(float(scale[0]), 1e-6))**2
            crop_area = float(box[2] * box[3] * crop_area_scale)
            sample.update(keypoints=joints.astype(np.float32), visible=visible,
                          person_box_area=crop_area)
            if self.annotation_areas is not None:
                sample['annotation_area'] = 0. if object_crop else float(self.annotation_areas[index] * crop_area_scale)
            if self.proposal_people is not None:
                sample.update(self.proposal_labels(index, image_id, center, scale,
                                                   image.shape[1] if flipped else 0, warp))
            if self.foreground_polygons is not None:
                visibility = target[FLIP, 2] if flipped else target[:, 2]
                sample.update(self.foreground_labels(index, warp, image.shape[1] if flipped else 0,
                                                     joints, visibility))
            if self.return_geometry:
                sample.update(warp_matrix=warp.astype(np.float32), flipped=flipped,
                              source_image_width=int(image.shape[1]))
        else:
            sample.update(bbox_score=float(self.scores[index]), area=float(np.prod(scale)))
        return sample


def preprocess(images, device, channels_last=False):
    x = images.to(device, non_blocking=True).float()[:, [2, 1, 0]]
    mean = x.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = x.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def merge_simcc_flip(original, mirrored):
    """Align a horizontally flipped crop and average the matching SimCC logits."""
    px, py = original
    fx, fy = mirrored
    return (px + fx[:, FLIP].flip(-1)) * .5, (py + fy[:, FLIP]) * .5


def simcc_targets(keypoints, visible, sigma_scale=1., continuous=False, annotation_area=None):
    mu = keypoints.float() * 2
    if not continuous:
        mu = torch.round(mu)
    sigma_x, sigma_y = mu.new_tensor([6. * sigma_scale, 6.93 * sigma_scale])
    if annotation_area is not None:
        # Use the median COCO train crop area while keeping widths near the pretrained codec.
        area_ratio = (annotation_area.float().clamp_min(1.) / 22000.).pow(.25)
        joint_ratio = (mu.new_tensor(SIGMAS) / .072).sqrt()
        relative_width = (area_ratio[:, None] * joint_ratio).clamp(.8, 1.2)
        sigma_x = sigma_x * relative_width
        sigma_y = sigma_y * relative_width
    x = torch.arange(WIDTH * 2, device=mu.device, dtype=torch.float32)
    y = torch.arange(HEIGHT * 2, device=mu.device, dtype=torch.float32)
    tx = torch.exp(-(x - mu[:, :, 0, None]).square() / (2 * sigma_x[..., None].square()))
    ty = torch.exp(-(y - mu[:, :, 1, None]).square() / (2 * sigma_y[..., None].square()))
    bounds = ((mu[:, :, 0] - 3*sigma_x < WIDTH*2) & (mu[:, :, 0] + 3*sigma_x+1 >= 0)
              & (mu[:, :, 1] - 3*sigma_y < HEIGHT*2) & (mu[:, :, 1] + 3*sigma_y+1 >= 0))
    return tx, ty, visible * bounds


def simcc_loss(outputs, keypoints, visible, reduction='per_sample', *,
               sigma_scale=1., continuous=False, hard_joint_weight=0., annotation_area=None):
    tx, ty, weights = simcc_targets(keypoints, visible, sigma_scale, continuous, annotation_area)
    if reduction not in {'per_sample', 'official'}:
        raise ValueError(reduction)
    loss, joint_loss = 0., 0.
    for pred, target in zip(outputs, (tx, ty)):
        bins = F.kl_div(F.log_softmax(pred.float()*10, -1), F.softmax(target*10, -1),
                        reduction='none')
        if reduction == 'official':
            # MMPose KLDiscretLoss: mean bins, sum local samples, divide joints.
            # Its scale therefore depends on the physical per-GPU minibatch.
            weighted = bins.mean(-1) * weights
            loss = loss + weighted.sum() / weights.shape[1]
        else:
            weighted = bins.sum(-1) * weights
            loss = loss + weighted.mean()
        if hard_joint_weight:
            joint_loss = joint_loss + weighted
    if hard_joint_weight:
        hard = joint_loss.topk(min(8, weights.shape[1]), dim=-1).values.mean()
        if reduction == 'official':
            hard = hard * weights.shape[0]
        loss = (1 - hard_joint_weight) * loss + hard_joint_weight * hard
    return loss


def simcc_log_expected_oks(outputs, keypoints, visible, person_area, kernel_scale):
    """Compute the per-joint log expected OKS on the two SimCC distributions."""
    points = keypoints.float()
    size = points.new_tensor((WIDTH, HEIGHT))
    valid = (torch.isfinite(points) & (points >= 0) & (points < size)).all(-1)
    weights = visible.float() * valid
    centers = torch.where(valid.unsqueeze(-1), points, 0.) * 2
    sigmas = points.new_tensor(SIGMAS)
    # COCO's spatial standard deviation is 2 sigma times the square root of area.
    std = ((4 * kernel_scale) * person_area.float().clamp_min(1.).sqrt()[:, None] * sigmas).clamp_min(1.)
    log_expected = 0.
    for axis, prediction in enumerate(outputs):
        bins = torch.arange(prediction.shape[-1], device=prediction.device, dtype=torch.float32)
        log_kernel = -.5 * ((bins - centers[..., axis, None]) / std[..., None]).square()
        log_probability = F.log_softmax(prediction.float()*10, -1)
        log_expected = log_expected + torch.logsumexp(log_probability + log_kernel, dim=-1)
    return log_expected, weights


def simcc_oks_loss(outputs, keypoints, visible, person_box_area):
    """Penalize individual localization errors with a half-width OKS kernel."""
    log_expected, weights = simcc_log_expected_oks(outputs, keypoints, visible, person_box_area,
                                                 kernel_scale=.5)
    return -(log_expected * weights).mean()


def simcc_pose_oks_loss(outputs, keypoints, visible, annotation_area):
    """Penalize poses whose mean expected OKS falls below the evaluation thresholds."""
    log_expected, weights = simcc_log_expected_oks(outputs, keypoints, visible, annotation_area,
                                                 kernel_scale=1.)
    joint_similarity = log_expected.exp()
    pose_similarity = (joint_similarity * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
    thresholds = torch.linspace(.5, .95, 10, device=pose_similarity.device)
    temperature = .03
    margin = (thresholds - pose_similarity[:, None]) / temperature
    shortfall = F.softplus(margin).mean(-1) * temperature
    return (shortfall * weights.amax(-1)).mean()


def pose_quality_loss(outputs, score_delta, keypoints, visible, person_box_area):
    """Fit joint confidence to agreement between the current prediction and GT."""
    with torch.no_grad():
        px, py = (axis.float() for axis in outputs)
        points = torch.stack((px.argmax(-1), py.argmax(-1)), dim=-1).float() / 2
        score = torch.minimum(px.amax(-1), py.amax(-1)).clamp_min(0.)
        gt = keypoints.float()
        valid = torch.isfinite(gt).all(-1)
        in_crop = valid & ((gt >= 0) & (gt < gt.new_tensor((WIDTH, HEIGHT)))).all(-1)
        distance = (points - torch.where(valid.unsqueeze(-1), gt, 0.)).square().sum(-1)
        std_squared = (2 * gt.new_tensor(SIGMAS)).square() * person_box_area.float().clamp_min(1.)[:, None]
        quality = torch.where(in_crop, torch.exp(-.5 * distance / std_squared), 0.)
    calibrated = score * score_delta.float().exp()
    return (F.smooth_l1_loss(calibrated, quality, beta=.1, reduction='none') * visible.float() * valid).mean()


def pose_oks_rank_loss(outputs, pose_delta, keypoints, visible, annotation_area, source):
    """Learn relative pose confidence from the original joint annotations."""
    with torch.no_grad():
        px, py = (axis.float() for axis in outputs)
        points = torch.stack((px.argmax(-1), py.argmax(-1)), dim=-1).float() * .5
        scores = torch.minimum(px.amax(-1), py.amax(-1))
        reliable = scores > .2
        raw = torch.where(reliable, scores, 0.).sum(-1) / reliable.sum(-1).clamp_min(1)

        valid = torch.isfinite(keypoints).all(-1) & (visible > 0)
        gt = torch.where(valid.unsqueeze(-1), keypoints.float(), 0.)
        distance_squared = (points - gt).square().sum(-1)
        variance = (2 * gt.new_tensor(SIGMAS)).square() * annotation_area.float().clamp_min(1.)[:, None]
        oks = (torch.exp(-.5 * distance_squared / variance) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        thresholds = torch.linspace(.5, .95, 10, device=gt.device)
        quality = torch.sigmoid((oks[:, None] - thresholds) / .025).mean(-1)
        target = torch.sigmoid((quality - quality.roll(1)) / .15)

        weights = valid.sum(-1).div(6.).clamp_max(1.) * torch.where(valid, visible.float(), 0.).amax(-1)
        weights = weights * (raw > 0)
        pair_weights = torch.minimum(weights, weights.roll(1)) * (source == source.roll(1))
        prior = raw.clamp_min(1e-6).log()

    correction = pose_delta.float().squeeze(-1)
    log_score = prior + correction
    logits = (log_score - log_score.roll(1)) / .1
    ranking = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (ranking * pair_weights).mean() + .1 * (correction.square() * weights).mean()


def pose_proposal_loss(outputs, logits, keypoints, areas, weights, hard_negatives=False):
    """Predict whether a model pose agrees with any annotated person in the crop."""
    with torch.no_grad():
        px, py = outputs
        points = torch.stack((px.argmax(-1), py.argmax(-1)), dim=-1).float() * .5
        peaks = torch.minimum(px.amax(-1), py.amax(-1)).float()
        detected = peaks > 0.
        points = torch.where(detected.unsqueeze(-1), points, -.5)
        valid = keypoints[..., 2] > 0
        distance = (points[:, None] - keypoints[..., :2]).square().sum(-1)
        variance = (2 * points.new_tensor(SIGMAS)).square() * areas[..., None].clamp_min(1.)
        similarities = (torch.exp(-.5 * distance / variance) * valid).sum(-1)
        best = (similarities / valid.sum(-1).clamp_min(1)).amax(-1)
        target = ((best - .15) / .35).clamp(0., 1.)
        if hard_negatives:
            confident = peaks > .2
            raw_score = (peaks * confident).sum(-1) / confident.sum(-1).clamp_min(1)
            emphasis = .5 + 3. * raw_score.clamp(0., 1.).square()
            weights = weights * torch.where(target < .5, emphasis, 1.)
    loss = F.binary_cross_entropy_with_logits(logits.float().squeeze(-1), target, reduction='none')
    return (loss * weights).mean()


def local_joint_loss(anchor, offset, keypoints, visible, max_shift):
    """Train sub-bin offsets only where the annotated point is within reach."""
    points = keypoints.float()
    target = points - anchor
    inside = ((points >= 0) & (points < points.new_tensor((WIDTH, HEIGHT)))).all(-1)
    nearby = torch.isfinite(target).all(-1) & (target.abs().amax(-1) < max_shift) & inside
    target = torch.where(nearby.unsqueeze(-1), target, 0.)
    error = F.smooth_l1_loss(offset.float(), target, reduction='none').mean(-1)
    return (error * visible.float() * nearby).mean()


def decode_batch(outputs, batch, score_delta=None):
    px, py = (x.float().cpu().numpy() for x in outputs)
    coords = np.stack((px.argmax(-1), py.argmax(-1)), -1).astype(np.float32) / 2
    scores = np.minimum(px.max(-1), py.max(-1))
    pose_scale = None
    if score_delta is not None:
        scale = np.exp(score_delta.float().cpu().numpy())
        if scale.shape == (len(scores), 1):
            pose_scale = scale[:, 0]
        else:
            scores = scores * scale
    coords[scores <= 0] = -.5
    center, scale = batch['center'].numpy(), batch['scale'].numpy()
    coords = coords / np.array([WIDTH, HEIGHT], np.float32) * scale[:, None] + center[:, None] - .5*scale[:, None]
    records = []
    for i in range(len(coords)):
        valid = scores[i] > .2
        score = float(batch['bbox_score'][i]) * (float(scores[i][valid].mean()) if valid.any() else 0.)
        if pose_scale is not None:
            score *= float(pose_scale[i])
        records.append(dict(image_id=int(batch['image_id'][i]), index=int(batch['index'][i]),
                            keypoints=np.concatenate((coords[i], scores[i, :, None]), -1),
                            score=score, area=float(batch['area'][i])))
    return records


def oks_overlap(reference, others):
    if not others:
        return np.array([])
    g = reference['keypoints']
    ds = np.stack([x['keypoints'] for x in others])
    d2 = np.sum((ds[:, :, :2] - g[None, :, :2]) ** 2, axis=-1)
    areas = (reference['area'] + np.array([x['area'] for x in others])) * .5
    e = d2 / ((SIGMAS * 2)**2)[None] / (areas[:, None] + np.spacing(1)) / 2
    return np.exp(-e).mean(-1)


def evaluate_records(records, output_dir, image_ids=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        if row['index'] in seen:
            raise ValueError('Duplicated evaluation sample')
        seen.add(row['index'])
        grouped[row['image_id']].append(row)
    results = []
    for image_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r:r['score'], reverse=True)
        while rows:
            row = rows.pop(0)
            results.append(dict(image_id=image_id, category_id=1,
                                keypoints=row['keypoints'].reshape(-1).tolist(), score=row['score']))
            overlap = oks_overlap(row, rows)
            rows = [r for r, o in zip(rows, overlap) if o <= .9]
    (output_dir / 'predictions.json').write_text(json.dumps(results))
    gt = COCO(str(DATA/'annotations/person_keypoints_val2017.json'))
    pred = gt.loadRes(results)
    evaluator = COCOeval(gt, pred, 'keypoints')
    if image_ids is not None:
        evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    metrics = {k:float(v) for k,v in zip(['AP','AP50','AP75','APm','APl','AR','AR50','AR75','ARm','ARl'], evaluator.stats)}
    metrics.update(samples=len(records), images=len(grouped), results=len(results))
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    prepare_records()
