# Train RSN + RTMPose-X

The release trains one model on COCO train2017 and MPII annotations. The complete 13-stage schedule is in [train.py](train.py). The published checkpoint is the EMA from the sixth epoch of the last confidence-calibration stage. [Model card](https://huggingface.co/CaiYuanhao/RSN-RTMPose-X)

## Environment and inputs

The reference runs used Python 3.11, PyTorch 2.13.0 with CUDA 12.6, and eight RTX A5000 GPUs. Run the commands from the RSN repository root on Linux.

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r projects/rsn_rtmpose/requirements-lock.txt
```

- Download the RTMPose-X 384×288 Body7 checkpoint. The training script verifies the official file and transparently extracts its tensor state. [OpenMMLab checkpoint](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth)
- Download COCO train2017 and val2017 images and `annotations_trainval2017.zip`. Put the image splits under the same COCO image directory. The archive provides the person labels and the non-person boxes used as hard negatives. [COCO downloads](https://cocodataset.org/#download)
- Download the original H56 person detection JSON, `COCO_val2017_detections_AP_H_56_person.json`. [MMPose dataset guide](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/docs/en/user_guides/prepare_datasets.md)
- Download the MPII images and RSN-format `train.json`. [RSN dataset instructions](../../README.md#dataset)

## Run

One invocation prepares the data cache, initializes the model, and runs each stage in order. The launcher starts distributed workers from the same `train.py`.

```bash
python -u projects/rsn_rtmpose/train.py \
  --checkpoint /data/downloads/rtmpose-x-official.pth \
  --coco-archive /data/downloads/annotations_trainval2017.zip \
  --coco-images /data/coco/images \
  --detections /data/coco/COCO_val2017_detections_AP_H_56_person.json \
  --mpii-annotations /data/mpii/train.json \
  --mpii-images /data/mpii/images \
  --data-dir /data/rsn-cache \
  --output /data/runs/rsn-rtmpose-x \
  --gpus 8
```

With an existing cache, pass `--data-dir` and omit the five raw dataset arguments. Add `--dry-run` to print all stage commands without starting training. Override `--batch-size` to fit your hardware. Per-GPU batch × number of GPUs must divide each stage's effective batch. On eight GPUs, 128 crops per GPU works with all three effective batches and uses accumulation of 4, 12, and 14 respectively. Actual memory needs depend on the GPU and stage.

The script uses FP16, activation checkpointing, gradient accumulation, distributed data parallel, and EMA. The handoff takes the weights from the recorded epoch with native validation. Each stage retains its learning-rate horizon, including when the handoff is earlier than the scheduled end.

| Stage | Crops/GPU × accumulation on 8 GPUs | Handoff epoch | Updates | Recorded AP |
| --- | ---: | ---: | ---: | ---: |
| Spatial heatmap refinement | 256 × 2 | 12 | 504 | 78.832509 |
| Wider RSN and pose head | 256 × 2 | 15 | 630 | 78.935134 |
| Proposal confidence | 512 × 1 | 8 | 336 | 78.981104 |
| Hard-negative proposals | 512 × 1 | 2 | 84 | 79.023650 |
| Proposal size | 512 × 1 | 5 | 210 | 79.046566 |
| Joint proposal confidence | 512 × 1 | 12 | 504 | 79.104194 |
| Joint geometry | 512 × 3 | 2 | 28 | 79.110166 |
| Wider spatial geometry | 256 × 6 | 1 | 14 | 79.112407 |
| Wider spatial confidence | 768 × 2 | 8 | 112 | 79.115303 |
| Spatial attention warmup | 512 × 3 | 3 | 42 | 79.114379 |
| Spatial attention geometry | 256 × 6 | 3 | 42 | 79.117887 |
| Spatial attention confidence | 512 × 3 | 8 | 112 | 79.123214 |
| Final SimCC confidence | 448 × 4 | 6 | 72 | **79.124422** |

The release uses a fixed-seed, fresh RSN branch on the official RTMPose-X core. The original spatial-heatmap stage also created a fresh RSN branch after the official core had been restored and frozen in the preceding experiment. The release has no teacher supervision. The table records earlier training runs. Differences in initialization, runtime and restart history can change the result of a rerun. [Stage record and published metrics](results/release.json)

## Resume and validate

Repeat the same command with the same directory. `pipeline.json` records the input and source hashes, stage ordering, completed validations, and handoff hashes. The active stage resumes from its latest saved epoch with its optimizer, AMP scaler and EMA. The finished release recipe writes `final_simcc_confidence/handoff.pth`. Intermediate handoffs record the validation lineage.

To validate the published checkpoint with the native evaluation protocol, use the same prepared data.

```bash
RSN_RTMPOSE_DATA=/data/rsn-cache python -m torch.distributed.run \
  --standalone --nproc-per-node=8 projects/rsn_rtmpose/evaluate.py \
  --checkpoint /data/models/rsn-rtmpose-x.pth \
  --output /data/runs/rsn-validation --batch-size 32 --workers 4 --amp fp16
```

The evaluator uses the full COCO val2017 split, H56 person proposals, flip inference and OKS NMS. The demo uses YOLOX-S detections.
