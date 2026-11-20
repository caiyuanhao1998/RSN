"""Rank and archive full-val EMA checkpoints."""
from contextlib import contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path


@contextmanager
def checkpoint_lock(run):
    with (Path(run)/'.checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def model_score(payload):
    metrics = payload.get('metrics', {})
    metrics = metrics.get('candidate', metrics)
    if (not metrics.get('full_validation') or metrics.get('samples') != 104125 or
        metrics.get('deployment', False) or metrics.get('amp', 'fp16') != 'fp16' or
        payload.get('trained_this_phase') is False):
        return None
    ap = metrics.get('AP')
    return float(ap) if ap is not None and math.isfinite(ap) else None


def archive_ranked_epoch(run, evaluation, keep):
    """Archive a qualifying non-record epoch from its matching atomic EMA save."""
    import torch

    if keep < 1 or not evaluation or evaluation.get('validation_variant'):
        return None
    metrics = evaluation['metrics']
    ap = model_score({'metrics': metrics})
    if ap is None or not metrics.get('flip'):
        return None
    run = Path(run)
    history = run/'best_history'
    epoch = evaluation['epoch']
    prefix = f'epoch_{epoch:03d}_AP{ap*100:.6f}_'
    resume = run/'last.pth'
    with checkpoint_lock(run):
        if not resume.is_file():
            return None
        existing = list(history.glob(prefix+'*.pth'))
        if existing:
            prune_epoch_archives(history, keep)
            return existing[0]
        saved = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
        if saved['epoch'] + 1 != epoch or saved['updates'] != evaluation['updates']:
            return None
        history.mkdir(exist_ok=True)
        temporary = history/f'.ranked_{epoch:03d}.{os.getpid()}.tmp'
        payload = dict(model=saved['ema'], args=saved['args'], epoch=epoch,
                       metrics=metrics, trained_this_phase=True)
        if 'seen_images' in saved:
            payload['ema_age_images'] = saved['seen_images'] + saved.get('ema_age_offset', 0)
        torch.save(payload, temporary)
        sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
        archive = history/f'{prefix}{sha256[:8]}.pth'
        temporary.replace(archive)
        prune_epoch_archives(history, keep)
        return archive



def prune_epoch_archives(history, keep):
    ranked = sorted(history.glob('epoch_*_AP*.pth'),
                    key=lambda path: float(path.stem.split('_AP')[1].split('_')[0]), reverse=True)
    for path in ranked[keep:]:
        path.unlink()
