# Local setup and runtime verification

Repository: `/home/ycai51/RSN`

Upstream revision: `975298f4980401babd32db33dac5ff96288eb66e`.

## Environment and storage

- Isolated Python environment: `/data/cyh23/rsn-repro/environment`.
- Python 3.11.7, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126.
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
  --extra-index-url https://download.pytorch.org/whl/cu126 \
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
- Kept MPII evaluation logs in plain numeric form with NumPy 2.

The model architectures and original full-training schedules are preserved. Training now accepts `--max-iters`, `--batch-size`, `--num-workers`, and `--output-dir` for bounded runtime checks.

## Reproduce checks

Run the regression suite:

```bash
cd /home/ycai51/RSN
python -m pytest -q tests
```

Check a model's CUDA forward/backward and finite gradients; repeat with `--efficient` for activation checkpointing:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/check_models.py --experiment RSN18.coco
```

The experiment folders `Res18.coco`, `RSN18.coco`, `4XRSN18.coco`, and `RSN18.mpii` sit at the repository root. Existing `exps/` paths resolve through symbolic links.

Run the real COCO training entrypoint and resume it:

```bash
cd /home/ycai51/RSN
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python RSN18.coco/train.py \
  --max-iters 40 --batch-size 32 --num-workers 4 \
  --output-dir /data/cyh23/rsn-repro/runs/RSN18.coco
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python RSN18.coco/train.py \
  --max-iters 45 --batch-size 32 --num-workers 4 \
  --output-dir /data/cyh23/rsn-repro/runs/RSN18.coco \
  -c /data/cyh23/rsn-repro/runs/RSN18.coco/iter-40.pth
```

Check two-GPU training:

```bash
cd /home/ycai51/RSN
CUDA_VISIBLE_DEVICES=5,7 OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=2 \
  RSN18.coco/train.py --max-iters 5 --batch-size 4 --num-workers 2 \
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

## Initial runtime verification results

- Downloaded and extracted all 82,783 COCO train2014 images, 40,504 COCO val2014 images, and 24,984 MPII images. The original image archives and extracted files are retained.
- Downloaded all seven annotation/detection files supplied by the author.
- Full training/validation image-reference audits passed with zero missing files for both datasets.
- All four model architectures passed CUDA forward/backward, finite-gradient, and output-shape checks in ordinary and activation-checkpointed modes (eight checks).
- RSN18 COCO trained for 40 real-data iterations at batch size 32, resumed to 45, and saved both checkpoints. Optimizer steps and scheduler epoch were verified as 40 and 45 respectively.
- RSN18 MPII trained for 20 real-data iterations at batch size 32 and saved its checkpoint.
- Res18 and 4XRSN18 each trained for five real-data COCO iterations and saved checkpoints.
- Two-GPU RSN18 COCO training completed five iterations. Variable-length NCCL prediction gathering also passed with the host transport setting above.
- Both datasets passed real-image flip inference with batches of two and one. COCOeval ran on the selected COCO image(s). The MPII evaluator separately passed a self-check using all 2,958 ground-truth validation poses.
- The initial nine regression tests passed, along with dependency consistency, syntax compilation, and `git diff --check`. The framework used for those initial runs has since been replaced by the security-updated environment described below.

Machine-readable results: `/data/cyh23/rsn-repro/validation/summary.json`.

Full logs: `/data/cyh23/rsn-repro/logs/`.

The MPII smoke check can be repeated with:

```bash
cd /home/ycai51/RSN
CUDA_VISIBLE_DEVICES=7 python scripts/check_data.py --experiment RSN18.mpii \
  --checkpoint /data/cyh23/rsn-repro/runs/RSN18.mpii/iter-20.pth \
  --output /data/cyh23/rsn-repro/validation/MPII
```

## Security upgrade

The active environment now uses PyTorch 2.13.0+cu126 and torchvision 0.28.0+cu126. The full CUDA, Triton, SymPy, and setuptools dependency set was reinstalled and the lock file regenerated. This resolves the PyTorch 2.5.1 vulnerability ranges reported by Dependabot, including [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) and [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).

The framework pair and CUDA build follow the [official installation matrix](https://pytorch.org/get-started/previous-versions/#v2130). Both runtime requirements and the complete lock file require the patched versions. All PyTorch checkpoint entrypoints explicitly request `weights_only=True`, and regression tests confirm Python objects outside the restricted checkpoint format are rejected. Load only checkpoints from trusted sources even with this restriction.

To audit the complete lock file without downloading/installing GPU wheels:

```bash
python -m pip install pip-audit==2.10.1
python scripts/audit_requirements.py requirements-lock.txt > "$TMPDIR/rsn-audit.txt"
python -m pip_audit --strict --no-deps --disable-pip -r "$TMPDIR/rsn-audit.txt"
```

The audit helper preserves every package and removes only PyTorch/torchvision CUDA build labels when looking up upstream vulnerability records. It never modifies the installable lock file. A prepared workflow template is available at `.github/workflow-templates/dependency-security.yml.example`. To enable automatic audits, copy it to `.github/workflows/dependency-security.yml` using a GitHub authorization that permits workflow changes. The current OAuth authorization lacks the `workflow` scope, so automatic Actions auditing is not enabled by this update; the local strict audit completed successfully.

New security-upgrade logs and reports are under `/data/cyh23/rsn-repro/security-upgrade/`. Earlier short-training checkpoints remain usable with the upgraded checkpoint loader.

Validation of the upgraded environment:

- 67 locked packages audited with zero known vulnerabilities; no dependencies or vulnerability IDs were ignored.
- 16 regression/security tests passed, including rejection of unsupported checkpoint Python objects and complete audit-input coverage.
- All four models passed CUDA forward/backward and inference checks in ordinary and efficient modes.
- Real COCO training completed 10 iterations and resumed to 15. A checkpoint produced under PyTorch 2.5.1 resumed from iteration 45 to 50 under PyTorch 2.13.
- Real MPII training completed 20 iterations; real-image inference and full image-reference checks passed.
- Two-GPU COCO training completed five iterations; variable-length NCCL prediction gathering passed.

The stable `/data/cyh23/rsn-repro/environment` path now points to `environment-torch213`. The earlier environment is retained as an inactive rollback copy at `environment-torch251-legacy`; it is excluded from the active execution path.
