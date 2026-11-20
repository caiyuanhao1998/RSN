"""Official-protocol COCO evaluation and matched native CUDA latency."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from models import RSN_PARAMETER_PREFIXES, RTMPoseX
from data_pipeline import ROOT, CocoPose, FLIP, preprocess, merge_simcc_flip, decode_batch, evaluate_records


def distributed_setup():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    torch.set_num_threads(4)
    return rank, local_rank, world


def load_model(path, variant=None, width=None, ema=True):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    args = saved.get('args', {})
    model = RTMPoseX(variant or args.get('variant', 'baseline'), width or args.get('width', 8),
                     num_steps=args.get('rsn_steps', 3),
                     lateral_channels=args.get('lateral_channels', 17),
                     spatial_stride=args.get('spatial_stride', 16),
                     context_width=args.get('context_width', 0), local_width=args.get('local_width', 0),
                     trunk_width=args.get('trunk_width', 0), spatial_width=args.get('spatial_width', 0),
                     pixel_width=args.get('pixel_width', 0), spatial_attention_width=args.get('spatial_attention_width', 0),
                     spatial_attention_layers=args.get('spatial_attention_layers', 1),
                     spatial_rsn_blocks=args.get('spatial_rsn_blocks', 0), spatial_rsn_width=args.get('spatial_rsn_width', 0),
                     spatial_rsn_attention_layers=args.get('spatial_rsn_attention_layers', 1))
    if args.get('pose_oks_confidence'):
        model.head.enable_pose_oks_confidence(proposal_confidence=args.get('proposal_confidence', False),
                                              proposal_size_confidence=args.get('proposal_size_confidence', False))
    model.head.raw_confidence_output = args.get('raw_confidence_output', False)
    state = saved.get('ema' if ema else 'model')
    if state is None:
        state = saved.get('state_dict', saved.get('model', saved))
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    assert all(k.startswith(RSN_PARAMETER_PREFIXES) for k in incompatible.missing_keys), incompatible.missing_keys
    return model


@torch.inference_mode()
def run_evaluation(model, output, batch_size=64, workers=4, flip=True, amp='fp16', limit=None, deployment=False,
                   dataset=None):
    rank, local_rank, world = distributed_setup()
    device = torch.device('cuda', local_rank)
    ds = CocoPose('val') if dataset is None else dataset
    use_proposal_size = getattr(model, 'uses_proposal_size', False)
    if use_proposal_size:
        ds.return_proposal_size = True
    n = len(ds) if limit is None else min(limit, len(ds))
    ds.indices = np.arange(rank, n, world)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=workers>0)
    model = model.to(device).eval()
    if deployment:
        assert amp == 'fp16'
        model = fuse_model(model).to(device=device, dtype=torch.float16)
        example = torch.zeros(batch_size, 3, 384, 288, device=device, dtype=torch.float16)
        metadata = (example.new_zeros((batch_size, 2)),) if use_proposal_size else ()
        runner = CUDAGraphRunner(model, example, *metadata)
    records = []
    start = time.perf_counter()
    dtype = {'fp16':torch.float16, 'bf16':torch.bfloat16}.get(amp)
    for iteration, batch in enumerate(loader):
        images = preprocess(batch['image'], device)
        metadata = (batch['proposal_size'].to(device, non_blocking=True),) if use_proposal_size else ()
        ctx = torch.autocast('cuda', dtype=dtype) if dtype else nullcontext()
        with ctx:
            count = images.shape[0]
            if deployment:
                images = images.half()
                if count < batch_size:
                    images = torch.nn.functional.pad(images, (0, 0, 0, 0, 0, 0, 0, batch_size-count))
                    metadata = tuple(torch.nn.functional.pad(item, (0, 0, 0, batch_size-count)) for item in metadata)
                outputs = tuple(x[:count].clone() for x in runner(images, *metadata))
            else:
                outputs = model(images, *metadata)
            px, py, *corrections = outputs
            delta = corrections[0] if corrections else None
            if flip:
                flipped = (tuple(x[:count] for x in runner(images.flip(-1), *metadata))
                           if deployment else model(images.flip(-1), *metadata))
                fx, fy, *flipped_corrections = flipped
                px, py = merge_simcc_flip((px, py), (fx, fy))
                if delta is not None:
                    flipped_delta = flipped_corrections[0]
                    if delta.shape[-1] == len(FLIP):
                        flipped_delta = flipped_delta[:, FLIP]
                    delta = (delta + flipped_delta)*.5
        records.extend(decode_batch((px, py), batch, score_delta=delta))
        if rank == 0 and (iteration + 1) % 50 == 0:
            print(f'eval {iteration+1}/{len(loader)} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    if world > 1:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(records, gathered, dst=0)
        records = [r for shard in gathered for r in shard] if rank == 0 else []
    metrics = None
    if rank == 0:
        assert len(records) == n
        image_ids = sorted({r['image_id'] for r in records}) if limit is not None else None
        metrics = evaluate_records(records, output, image_ids=image_ids)
        metrics.update(flip=flip, amp=amp, seconds=time.perf_counter()-start,
                       full_validation=limit is None, deployment=deployment,
                       bbox_protocol=getattr(ds, 'validation_protocol', 'person_detections_H56'))
        if hasattr(ds, 'validation_record_sha256'):
            metrics['validation_record_sha256'] = ds.validation_record_sha256
        (Path(output) / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    if world > 1:
        dist.barrier()
    return metrics


def fuse_model(model):
    from models import ConvBNAct
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if isinstance(module, ConvBNAct):
            module.conv = fuse_conv_bn_eval(module.conv, module.bn)
            module.bn = torch.nn.Identity()
    return model


class CUDAGraphRunner:
    """Fixed-shape deployment runner; replay updates both input and output.

    Returned tensors are reused on the next call. Consume them immediately or
    clone them when storing multiple requests. Both baseline and fusion use the
    same runner and FP16 Conv/BN fusion in the deployment speed comparison.
    """
    def __init__(self, model, example, *metadata):
        self.model = model
        self.input = example.clone()
        self.metadata = tuple(value.clone() for value in metadata)
        for _ in range(20): model(self.input, *self.metadata)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = model(self.input, *self.metadata)

    def __call__(self, images, *metadata):
        if (images.shape != self.input.shape or len(metadata) != len(self.metadata) or
                any(x.shape != y.shape for x, y in zip(metadata, self.metadata))):
            raise ValueError('CUDA Graph runner requires its captured input shape')
        self.input.copy_(images)
        for current, source in zip(self.metadata, metadata):
            current.copy_(source)
        self.graph.replay()
        return self.outputs


@torch.inference_mode()
def benchmark(models, output, warmup=50, iterations=300, rounds=3, cuda_graph=False, paired=False):
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    x = torch.randn(1, 3, 384, 288, device=device, dtype=torch.float16)
    backend = ('PyTorch CUDA Graph FP16' if cuda_graph else 'PyTorch CUDA FP16') + ', fused Conv/BN, batch=1, no flip'

    def prepare(model):
        model = fuse_model(model).to(device=device, dtype=torch.float16)
        metadata = (x.new_zeros((1, 2)),) if getattr(model, 'uses_proposal_size', False) else ()
        for _ in range(warmup):
            model(x, *metadata)
        torch.cuda.synchronize()
        runner = CUDAGraphRunner(model, x, *metadata) if cuda_graph else model
        return runner, metadata, sum(p.numel() for p in model.parameters())

    def measure(entry):
        runner, metadata, _ = entry
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            runner(x, *metadata)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)/iterations

    def summarize(name, entry, times):
        return dict(name=name, params=entry[2], latency_ms=float(np.median(times)), rounds_ms=times,
                    backend=backend, device=torch.cuda.get_device_name(0))

    results = []
    if paired:
        (reference_name, reference_model), *candidates = models
        reference = prepare(reference_model)
        reference_times = []
        for name, model in candidates:
            candidate = prepare(model)
            timings = [[], []]
            entries = reference, candidate
            for round_index in range(rounds):
                # Alternate the order so a change in GPU load affects both models.
                order = (0, 1) if round_index % 2 == 0 else (1, 0)
                for index in order:
                    timings[index].append(measure(entries[index]))
            reference_times.extend(timings[0])
            row = summarize(name, candidate, timings[1])
            baseline = float(np.median(timings[0]))
            row.update(latency_ratio=row['latency_ms']/baseline,
                       paired_reference_ms=baseline, paired_reference_rounds_ms=timings[0])
            results.append(row)
            del entries, candidate
            torch.cuda.empty_cache()
        if not reference_times:
            reference_times = [measure(reference) for _ in range(rounds)]
        results.insert(0, dict(summarize(reference_name, reference, reference_times), latency_ratio=1.))
    else:
        for name, model in models:
            entry = prepare(model)
            results.append(summarize(name, entry, [measure(entry) for _ in range(rounds)]))
            del entry
            torch.cuda.empty_cache()
        baseline = results[0]['latency_ms']
        for row in results:
            row['latency_ratio'] = row['latency_ms']/baseline
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/rtmpose-x-state-only.pth')
    parser.add_argument('--variant', choices=['baseline','rsn','rsn_fusion','rsn_highres'], default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', choices=['fp16','bf16','fp32'], default='fp16')
    parser.add_argument('--no-flip', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--reference', type=Path, help='Tensor-only official weights for a paired benchmark')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--deployment-eval', action='store_true')
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.variant, args.width)
    if args.benchmark:
        if args.reference is None:
            parser.error('--benchmark requires --reference')
        entries = [('official', load_model(args.reference)), ('candidate', model)]
        benchmark(entries, args.output, cuda_graph=args.cuda_graph, paired=True)
    else:
        run_evaluation(model, args.output, args.batch_size, args.workers,
                       not args.no_flip, args.amp, args.limit, args.deployment_eval)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__': main()
