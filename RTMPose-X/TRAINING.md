# Train RSN + RTMPose-X

Use Python 3.11. From the repository root, install the [locked training dependencies](requirements-lock.txt) with the CUDA 12.6 wheel index.

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r RTMPose-X/requirements-lock.txt
```

- Use the [official RTMPose-X Body7 checkpoint](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth).
- Set up the COCO and MPII image folders and MPII `train.json` as described in [Dataset](../README.md#dataset).
- Download `COCO_val2017_detections_AP_H_56_person.json` for validation. [MMPose dataset guide](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/docs/en/user_guides/prepare_datasets.md)
- Download `annotations_trainval2017.zip` for the 2017 COCO training and validation labels. [COCO downloads](https://cocodataset.org/#download)

Run from the repository root.

```bash
python RTMPose-X/train.py --checkpoint /path/to/rtmpose-x-body7.pth \
  --coco-archive /path/to/annotations_trainval2017.zip --coco-images dataset/COCO/images \
  --detections /path/to/COCO_val2017_detections_AP_H_56_person.json \
  --mpii-annotations dataset/MPII/gt_json/train.json --mpii-images dataset/MPII/images \
  --output model_logs/RTMPose-X --gpus 8
```

The data cache defaults to `RTMPose-X/data`. Repeat the command with the same paths to resume.
