"""COCO and MPII training with explicitly masked COCO-format joint labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data_pipeline import DATA, CocoPose

# Same twelve common limb correspondences as the pinned MMPose cocktail recipe.
MPII_TO_COCO = [(0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15),
                (10, 10), (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)]
MPII_ROOT = Path('/mnt/realccvl15/ycai51/RSN-data')


def convert_mpii_annotation(annotation):
    """Convert this RSN dataset's MATLAB coordinates and unpadded body box."""
    center = np.asarray(annotation['center'], dtype=np.float32).copy()
    scalar = float(annotation['scale'])
    if center[0] != -1:
        center[1] += 15 * scalar
    center -= 1
    scale = np.full(2, 200 * scalar, dtype=np.float32)
    box = np.concatenate((center - scale/2, scale))
    source = np.asarray(annotation['joints'], dtype=np.float32) - 1
    visible = np.asarray(annotation['joints_vis'], dtype=np.float32) > 0
    joints = np.zeros((17, 3), dtype=np.float32)
    for original, target in MPII_TO_COCO:
        joints[target, :2] = source[original]
        joints[target, 2] = visible[original]
    # No artificial nose/eye/ear labels are inferred from MPII's head/neck.
    return box, joints


def prepare_mpii(annotation_path, image_root, target=DATA):
    target = Path(target)
    image_root = Path(image_root).resolve()
    annotations = json.loads(annotation_path.read_text())
    ids, boxes, keypoints, image_index = [], [], [], {}
    for annotation in annotations:
        box, joints = convert_mpii_annotation(annotation)
        if not joints[:, 2].any() or np.any(box[2:] <= 0):
            continue
        path = image_root/annotation['image']
        # Negative IDs keep MPII and COCO namespaces disjoint.
        image_id = -int(path.stem)-1
        image_index[str(image_id)] = str(path)
        ids.append(image_id); boxes.append(box); keypoints.append(joints)
    missing = [p for p in image_index.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} MPII images; first: {missing[0]}')
    np.savez(target/'mpii_train_records.npz', image_ids=np.array(ids, np.int64),
             boxes=np.array(boxes, np.float32), keypoints=np.array(keypoints, np.float32))
    (target/'mpii_image_index.json').write_text(json.dumps(image_index))
    report = dict(annotation_file=str(annotation_path),
                  annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                  people=len(ids), images=len(image_index), keypoint_mapping=MPII_TO_COCO,
                  facial_keypoints_supervised=False, validation_annotations_used=False,
                  coordinates='RSN source joints and centers converted from MATLAB 1-based coordinates',
                  bbox='Center y += 15*scale, unpadded side=200*scale; shared crop then pads by 1.25')
    (target/'mpii_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


class CocoMpiiPose(CocoPose):
    def __init__(self, augment='finetune'):
        super().__init__('train', augment=augment)
        count = len(self.ids)
        arrays = np.load(DATA/'mpii_train_records.npz')
        assert np.all(self.ids >= 0) and np.all(arrays['image_ids'] < 0)
        self.ids = np.concatenate((self.ids, arrays['image_ids']))
        self.boxes = np.concatenate((self.boxes, arrays['boxes']))
        self.keypoints = np.concatenate((self.keypoints, arrays['keypoints']))
        self.image_index.update(json.loads((DATA/'mpii_image_index.json').read_text()))
        self.sources = np.concatenate((np.zeros(count, np.uint8),
                                      np.ones(len(arrays['image_ids']), np.uint8)))
        self.indices = np.arange(len(self.ids))
        self.source_counts = dict(COCO=count, MPII=len(arrays['image_ids']))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['source'] = int(self.sources[self.indices[index]])
        return sample


class CocoOnlyPose(CocoPose):
    """Use the same COCO pipeline for the sequential data-source ablation."""
    def __init__(self, augment='finetune'):
        super().__init__('train', augment=augment)
        self.source_counts = dict(COCO=len(self.ids), MPII=0)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['source'] = 0
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path,
                        default=MPII_ROOT/'annotations/MPII/gt_json/train.json')
    parser.add_argument('--image-root', type=Path, default=MPII_ROOT/'MPII/images')
    options = parser.parse_args()
    prepare_mpii(options.annotation, options.image_root)
