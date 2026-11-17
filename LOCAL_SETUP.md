# Local setup and runtime verification

Repository: `/home/ycai51/RSN`

Upstream revision: `975298f4980401babd32db33dac5ff96288eb66e`.

## Environment and storage

- Isolated Python environment: `/data/cyh23/rsn-repro/environment`.
- Python 3.11.7, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121.
- NumPy 1.26.4, OpenCV headless 4.11.0, scipy 1.17.1, pycocotools 2.0.11.
- Exact installed package versions: `requirements-lock.txt`.
- Dataset storage: `/mnt/realccvl15/ycai51/RSN-data`.
- Runtime logs, checkpoints, validation reports: `/data/cyh23/rsn-repro`.
- Repository dataset directories and `model_logs` link to those locations.
- Dataset sources, archive sizes, and annotation SHA-256 values: `/mnt/realccvl15/ycai51/RSN-data/manifest.json`.

Activate the project environment in Bash:

```bash
source /home/ycai51/RSN/scripts/env.sh
```

This sets `RSN_HOME`, `PYTHONPATH`, the environment's executable path, and temporary/cache directories. On the verified host, the ignored `scripts/env.local.sh` selects the storage paths above and defaults `NCCL_P2P_DISABLE=1`: direct GPU peer transport stalled on this host; two-GPU NCCL communication passed using shared-memory transport. This setting is confined to the shell that sources this file.

On another machine, `scripts/env.sh` defaults to an environment at `.venv` and runtime files at `.runtime` within the repository. Set `RSN_ENV_DIR` and `RSN_RUNTIME_DIR` before sourcing it to choose other locations, or put those exports in an ignored `scripts/env.local.sh`. NCCL transport is unchanged unless your own configuration sets it.

To reproduce the Python environment in a new destination, first create a Python 3.11 virtual environment, then use its pip:

```bash
TMPDIR=/data/cyh23/rsn-repro/tmp python -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r /home/ycai51/RSN/requirements-lock.txt
```

## Fixes

- Added omitted runtime dependencies to `requirements.txt`.
- Corrected the README's nonexistent `RSN50.coco` example directory to `RSN18.coco`.
- Fixed single-process rank initialization and selected CUDA device handling.
- Accepted both local-rank argument spellings and the `LOCAL_RANK` environment variable used by `torchrun`.
- Created output directories before opening log files; fixed the engine's default log filename.
- Fixed the Res18 training/evaluation imports to use its actual `MSPN` model class.
- Preserved the batch dimension during COCO confidence reduction when the final batch has one example.
- Restored the sampler's starting iteration during resume and stepped the LR scheduler after the optimizer.
- Limited TensorBoard/checkpoint writes to the global primary process and closed writers/process groups on normal completion.
- Fixed the SGD optimizer configuration lookup and small-dataset distributed sampler padding.
- Kept COCO keypoints as floating-point coordinates during affine augmentation; corrected the excluded-image ID's type.
- Made activation checkpointing's `use_reentrant` choice explicit for current PyTorch.
- Made missing evaluation checkpoints fail explicitly instead of evaluating random weights silently.
- Corrected MPII `Mean@0.1` to use threshold 0.10 rather than 0.11, with a boundary regression test.

The model architectures and original full-training schedules are preserved. Training now accepts `--max-iters`, `--batch-size`, `--num-workers`, and `--output-dir` for bounded runtime checks.

## Reproduce checks

Run the regression suite:

```bash
cd /home/ycai51/RSN
python -m pytest -q tests/test_runtime.py
```

Check a model's CUDA forward/backward and finite gradients; repeat with `--efficient` for activation checkpointing:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/check_models.py --experiment RSN18.coco
```

Supported experiments: `Res18.coco`, `RSN18.coco`, `4XRSN18.coco`, and `RSN18.mpii`.

Run the real COCO training entrypoint and resume it:

```bash
cd /home/ycai51/RSN/exps/RSN18.coco
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python train.py \
  --max-iters 40 --batch-size 32 --num-workers 4 \
  --output-dir /data/cyh23/rsn-repro/runs/RSN18.coco
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python train.py \
  --max-iters 45 --batch-size 32 --num-workers 4 \
  --output-dir /data/cyh23/rsn-repro/runs/RSN18.coco \
  -c /data/cyh23/rsn-repro/runs/RSN18.coco/iter-40.pth
```

Check two-GPU training:

```bash
cd /home/ycai51/RSN
CUDA_VISIBLE_DEVICES=5,7 OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=2 \
  exps/RSN18.coco/train.py --max-iters 5 --batch-size 4 --num-workers 2 \
  --output-dir /data/cyh23/rsn-repro/runs/RSN18.coco-ddp
```

Audit every dataset image reference and run flip inference on three real examples, including a final batch of one:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/check_data.py --experiment RSN18.coco \
  --checkpoint /data/cyh23/rsn-repro/runs/RSN18.coco/iter-40.pth \
  --output /data/cyh23/rsn-repro/validation/COCO
```

The COCO check also runs COCOeval on the selected image(s). These bounded checks verify runtime behavior. Their metrics are not converged-model accuracy or reproduction of paper results.

For normal full training, omit `--max-iters` and follow the README. For the standard evaluation entrypoint, use a checkpoint in the experiment's configured `OUTPUT_DIR`; it now reports an error if that checkpoint is missing.

## Completed verification results

- Downloaded and extracted all 82,783 COCO train2014 images, 40,504 COCO val2014 images, and 24,984 MPII images. The original image archives and extracted files are retained.
- Downloaded all seven annotation/detection files supplied by the author.
- Full training/validation image-reference audits passed with zero missing files for both datasets.
- All four model architectures passed CUDA forward/backward, finite-gradient, and output-shape checks in ordinary and activation-checkpointed modes (eight checks).
- RSN18 COCO trained for 40 real-data iterations at batch size 32, resumed to 45, and saved both checkpoints. Optimizer steps and scheduler epoch were verified as 40 and 45 respectively.
- RSN18 MPII trained for 20 real-data iterations at batch size 32 and saved its checkpoint.
- Res18 and 4XRSN18 each trained for five real-data COCO iterations and saved checkpoints.
- Two-GPU RSN18 COCO training completed five iterations. Variable-length NCCL prediction gathering also passed with the host transport setting above.
- Both datasets passed real-image flip inference with batches of two and one. COCOeval ran on the selected COCO image(s). The MPII evaluator separately passed a self-check using all 2,958 ground-truth validation poses.
- Nine regression tests passed. Dependency consistency, syntax compilation, and `git diff --check` passed. PyTorch 2.5 emits a non-failing future warning for the original checkpoint-loading default.

Machine-readable results: `/data/cyh23/rsn-repro/validation/summary.json`.

Full logs: `/data/cyh23/rsn-repro/logs/`.

The MPII smoke check can be repeated with:

```bash
cd /home/ycai51/RSN
CUDA_VISIBLE_DEVICES=7 python scripts/check_data.py --experiment RSN18.mpii \
  --checkpoint /data/cyh23/rsn-repro/runs/RSN18.mpii/iter-20.pth \
  --output /data/cyh23/rsn-repro/validation/MPII
```
