"""
@author: Yuanhao Cai
@date:  2020.03
"""

import cv2
import json
import numpy as np
import os
from scipy.io import loadmat
from collections import OrderedDict

from dataset.JointsDataset import JointsDataset

class MPIIDataset(JointsDataset):

    def __init__(self, DATASET, stage, transform=None):
        super().__init__(DATASET, stage, transform)
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]

        self.train_gt_file = 'train.json'
        self.train_gt_path = os.path.join(self.cur_dir, 'gt_json',
                self.train_gt_file)

        self.val_gt_file = 'valid.json'
        self.val_gt_path = os.path.join(self.cur_dir, 'gt_json',
                self.val_gt_file)
        self.val_gt_mat = os.path.join(self.cur_dir, 'gt_json', 'valid.mat')

        self.test_det_file = 'test.json'
        self.test_det_path = os.path.join(self.cur_dir, 'det_json',
                self.test_det_file)

        self.data = self._get_data()
        self.data_num = len(self.data)

    def _get_data(self):
        data = list()

        if self.stage == 'train':
            mpii = json.load(open(self.train_gt_path))
        elif self.stage == 'val':
            mpii = json.load(open(self.val_gt_path))
        else:
            mpii = json.load(open(self.test_det_path))

        for d in mpii:
            img_name = d['image']
            img_id = img_name.split('.')[0]
            img_path = os.path.join(self.cur_dir, 'images', img_name)

            center = np.array(d['center'], dtype=np.float32)
            scale = np.array([d['scale'], d['scale']], dtype=np.float32)

            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
            center -= 1

            if self.stage == 'test':
                joints = np.zeros((self.keypoint_num, 3), dtype=np.float32)
            else:
                joints = np.array(d['joints'], dtype=np.float32)
                joints -= 1
                joints_vis = np.array(d['joints_vis'], dtype=np.float32)
                joints_vis = joints_vis.reshape(-1, 1) * 2
                joints = np.concatenate((joints, joints_vis), axis=1)

            data.append(dict(center=center,
                             img_id=img_id,
                             img_path=img_path,
                             img_name=img_name,
                             joints=joints,
                             scale=scale))

        return data

    # referring msra high resolution
    def evaluate(self, preds):
        preds = preds[:, :, 0:2] + 1.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.val_gt_mat)
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        print(name_value)

    def visualize(self, img, joints, score=None):
        pairs = [[0, 1], [1, 2], [2, 6], [3, 4], [3, 6], [4, 5], [6, 7],
                 [7, 8], [8, 9], [8, 12], [8, 13], [10, 11], [11, 12],
                 [13, 14], [14, 15]]
        color = np.random.randint(0, 256, (self.keypoint_num, 3)).tolist()

        for i in range(self.keypoint_num):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                cv2.circle(img, tuple(joints[i, :2]), 2, tuple(color[i]), 2)
        if score:
            cv2.putText(img, score, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (128, 255, 0), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints[pair[0] - 1], joints[pair[1] - 1])

        return img
            

if __name__ == '__main__':
    from dataset.attribute import load_dataset
    dataset = load_dataset('MPII')
    mpii = MPIIDataset(dataset, 'val')
    print(mpii.data_num)

