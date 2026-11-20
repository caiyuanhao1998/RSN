# RSN + RTMPose-X

A top-down multi-person pose model with a high-resolution residual-step refiner, pooled spatial attention and proposal-conditioned confidence. It predicts 17 COCO joints from 384 × 288 person crops.

[Model and checkpoint](https://huggingface.co/CaiYuanhao/RSN-RTMPose-X) · [Online demo](https://huggingface.co/spaces/CaiYuanhao/RSN-RTMPose-X) · [Project page](https://caiyuanhao1998.github.io/RSN-Project/) · [RSN paper](https://arxiv.org/abs/2003.04030)

| Model | COCO val2017 AP | Pose network latency |
| --- | ---: | ---: |
| Official RTMPose-X | 78.818237 | 3.338 ms |
| RSN + RTMPose-X | **79.124422** | 5.401 ms |

Native AP uses the full COCO val2017 split, H56 proposals, flip inference and OKS NMS. Latency uses paired single-person FP16 CUDA Graph runs with fused Conv/BN on the same RTX A5000. [Recorded metrics](results/release.json)

## Inference

Use a Python 3.11 environment. The demo requirements install CPU PyTorch. In a CUDA training environment, keep the CUDA build and install Gradio, Hugging Face Hub and imageio-ffmpeg alongside the training requirements.

```bash
python -m pip install -r projects/rsn_rtmpose/requirements-demo.txt
python projects/rsn_rtmpose/pipeline.py input.jpg poses.jpg
python projects/rsn_rtmpose/pipeline.py input.mp4 poses.mp4
python projects/rsn_rtmpose/app.py
```

The pipeline downloads the published weights and official YOLOX-S ONNX detector on first use. It batches person crops and selects CUDA for pose inference when available. Add `--device cpu`, `--show-boxes`, or `--flip` as needed. The video CLI streams frames to a downloadable H.264 MP4. The pose overlay is computed independently for each frame. [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

The Python interface takes OpenCV BGR frames and returns the bounding box, detection confidence, and scored COCO joints for each person.

```python
import cv2
from pipeline import PosePipeline, draw_poses

pipeline = PosePipeline()
frame = cv2.imread("input.jpg")
poses = pipeline.predict(frame)
cv2.imwrite("poses.jpg", draw_poses(frame, poses))
```

Run the Python snippet from `projects/rsn_rtmpose` or add that directory to `PYTHONPATH`.

## Training and media

Use one training entry point for data preparation, initialization and all sequential COCO + MPII stages. [Training guide](TRAINING.md)

See the original source credits and downloadable pose overlays. [Sports videos and GIFs](MEDIA.md)
