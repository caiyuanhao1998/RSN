# RSN + RTMPose-X

A top-down multi-person pose model with a high-resolution residual-step refiner, pooled spatial attention and proposal-conditioned confidence. It predicts 17 COCO joints from 384 × 288 person crops.

[Model and checkpoint](https://huggingface.co/CaiYuanhao/RSN) · [Online demo](https://huggingface.co/spaces/CaiYuanhao/RSN) · [Project page](https://caiyuanhao1998.github.io/RSN-Project/) · [RSN paper](https://arxiv.org/abs/2003.04030)

## COCO val2017

| Model | AP | AP50 | AP75 | AR |
| --- | ---: | ---: | ---: | ---: |
| RTMPose-X (reproduced) | 78.818 | 91.475 | 85.176 | 83.264 |
| RSN + RTMPose-X | **79.124** | **91.586** | **85.652** | **83.492** |
| Gain (points) | +0.306 | +0.111 | +0.476 | +0.228 |

Both models use the full COCO val2017 split with H56 person detections, flip inference, and OKS NMS. [Baseline metrics](results/baseline_official.json) · [Release metrics](results/release.json)

## Inference

Use a Python 3.11 environment. The demo requirements install CPU PyTorch. In a CUDA training environment, keep the CUDA build and install Gradio, Hugging Face Hub and imageio-ffmpeg alongside the training requirements.

```bash
python -m pip install -r RTMPose-X/requirements-demo.txt
python RTMPose-X/pipeline.py input.jpg poses.jpg
python RTMPose-X/pipeline.py input.mp4 poses.mp4
python RTMPose-X/app.py
```

The pipeline downloads the published weights and official YOLOX-S ONNX detector on first use. It batches person crops and selects CUDA for pose inference when available. Add `--device cpu`, `--show-boxes`, or `--flip` as needed. The video CLI streams frames to a downloadable H.264 MP4. The pose overlay is computed independently for each frame. [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

The demo header shows completed image and video estimates for today and the last 30 calendar days in Pacific time. For a self-hosted demo, set `RSN_USAGE_DIR` to an existing writable, persistent directory. The Space mounts a private bucket at `/mnt/rsn-usage` and stores one anonymous marker per completed estimate.

The Python interface takes OpenCV BGR frames and returns the bounding box, detection confidence, and scored COCO joints for each person.

```python
import cv2
from pipeline import PosePipeline, draw_poses

pipeline = PosePipeline()
frame = cv2.imread("input.jpg")
poses = pipeline.predict(frame)
cv2.imwrite("poses.jpg", draw_poses(frame, poses))
```

Run the Python snippet from `RTMPose-X` or add that directory to `PYTHONPATH`.

## Training

Follow [Training in the main README](../README.md#4-training) for COCO and MPII.

## Evaluation

Run from the repository root with the prepared data cache and the published checkpoint.

```bash
RSN_RTMPOSE_DATA=RTMPose-X/data torchrun --standalone --nproc-per-node=8 \
  RTMPose-X/evaluate.py --checkpoint /path/to/rsn-rtmpose-x.pth \
  --output model_logs/rtmpose-x-validation --batch-size 32 --workers 4 --amp fp16
```

## Media

[Sports videos, GIFs and footage credits](MEDIA.md)
