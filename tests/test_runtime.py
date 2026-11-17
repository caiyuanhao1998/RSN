"""Regression tests for training setup, resume and last-batch inference."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('RSN_HOME', str(ROOT))
sys.path.insert(0, str(ROOT))


def load_experiment(name, module):
    directory = ROOT / 'exps' / name
    previous = {key: sys.modules.pop(key, None) for key in ('config', 'network')}
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(f'{name}_{module}', directory / f'{module}.py')
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    finally:
        sys.path.pop(0)
        for key, value in previous.items():
            sys.modules.pop(key, None)
            if value is not None:
                sys.modules[key] = value


def test_single_process_engine_ranks(monkeypatch):
    from cvpack.torch_modeling.engine.engine import Engine
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, 'argv', ['train.py'])
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    engine = Engine(EasyDict())
    assert engine.local_rank == 0
    assert engine.world_rank == 0
    assert engine.world_size == 1


def test_logger_creates_output_directory(tmp_path):
    from cvpack.utils.logger import get_logger
    output = tmp_path / 'new' / 'experiment'
    logger = get_logger('test_runtime_directory', str(output))
    logger.info('created')
    assert (output / 'log.txt').is_file()


def test_sgd_optimizer():
    from lib.utils.solver import make_optimizer
    cfg = EasyDict(SOLVER=dict(OPTIMIZER='SGD', BASE_LR=0.01, MOMENTUM=0.9, WEIGHT_DECAY=0))
    optimizer = make_optimizer(cfg, torch.nn.Linear(2, 1), 1)
    assert isinstance(optimizer, torch.optim.SGD)


@pytest.mark.parametrize('experiment', ['Res18.coco', 'RSN18.coco', '4XRSN18.coco'])
def test_coco_inference_batch_of_one(experiment):
    module = load_experiment(experiment, 'test')
    class Model(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(len(images), 17, 64, 48)
    batch = (torch.zeros(1, 3, 256, 192), [1.0],
             [np.array([100., 100.])], [np.array([1., 1.])], [1])
    result = module.compute_on_dataset(Model(), [batch], torch.device('cpu'))
    assert len(result) == 1
    assert len(result[0]['keypoints']) == 51


def test_distributed_sampler_tiny_dataset():
    from cvpack.dataset.torch_samplers.distributed import DistributedSampler
    assert list(DistributedSampler([0], num_replicas=4, rank=3)) == [0]


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    from cvpack.torch_modeling.engine.engine import Engine
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, 'argv', ['train.py'])
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    engine = Engine(EasyDict())
    engine.setup_log('test_checkpoint_round_trip', str(tmp_path))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    engine.register_state(model=model, optimizer=optimizer, scheduler=scheduler, iteration=2)
    checkpoint = tmp_path / 'checkpoint.pth'
    engine.save_checkpoint(str(checkpoint))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    engine.state.iteration = 0
    engine.load_checkpoint(str(checkpoint), is_restore=False)
    assert engine.state.iteration == 2
    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.0001)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    assert all(int(state['step']) == 2 for state in optimizer.state.values())


def test_mpii_pckh_point_one_threshold(monkeypatch, capsys):
    import dataset.MPII.mpii as module
    names = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelvis',
             'thorax', 'neck', 'head', 'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']
    ground_truth = dict(dataset_joints=np.array([names]),
                        jnt_missing=np.zeros((16, 1)),
                        pos_gt_src=np.zeros((16, 2, 1)),
                        headboxes_src=np.array([[[0.], [0.]], [[1.], [0.]]]))
    monkeypatch.setattr(module, 'loadmat', lambda _: ground_truth)
    dataset = module.MPIIDataset.__new__(module.MPIIDataset)
    dataset.val_gt_mat = 'valid.mat'
    # Normalized error 0.105 must fail at 0.1 and pass at 0.11.
    predictions = np.full((1, 16, 3), -1.)
    predictions[:, :, 0] += 0.105 * 0.6
    dataset.evaluate(predictions)
    assert "('Mean@0.1', 0.0)" in capsys.readouterr().out


class UnsupportedCheckpointMetadata:
    """Ordinary Python objects must not be unpickled by checkpoint loaders."""


@pytest.mark.parametrize('loader', ['engine', 'model'])
def test_checkpoint_rejects_python_objects(loader, monkeypatch, tmp_path):
    import logging
    import pickle
    from cvpack.torch_modeling.engine.engine import Engine
    from cvpack.torch_modeling.engine.checkpoint import load_model

    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / 'python-object.pth'
    torch.save({'model': model.state_dict(),
                'metadata': UnsupportedCheckpointMetadata()}, checkpoint)
    if loader == 'engine':
        for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(sys, 'argv', ['train.py'])
        monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
        monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
        engine = Engine(EasyDict())
        engine.register_state(model=model)
        engine.setup_log('test_restricted_checkpoint', str(tmp_path))
        with pytest.raises(pickle.UnpicklingError):
            engine.load_checkpoint(str(checkpoint))
    else:
        with pytest.raises(pickle.UnpicklingError):
            load_model(model, str(checkpoint), logging.getLogger('restricted_model'))
