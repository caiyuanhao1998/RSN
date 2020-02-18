# encoding: utf-8

import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from collections import OrderedDict
from cvpack.utils.pyt_utils import (
    parse_torch_devices, extant_file, link_file, ensure_dir)
from cvpack.utils.logger import get_logger

from .checkpoint import load_model


class State(object):
    def __init__(self):
        self.iteration = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['iteration', 'model', 'optimizer', 'scheduler']
            setattr(self, k, v)


class Engine(object):
    def __init__(self, cfg, custom_parser=None):
        self.version = 0.01
        self.state = State()
        self.devices = None
        self.distributed = False
        self.logger = None
        self.cfg = cfg

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.world_rank = int(os.environ['RANK'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            dist.barrier()
            self.devices = [i for i in range(self.world_size)]
        else:
            # todo check non-distributed training
            self.world_rank = 1
            self.devices = parse_torch_devices(self.args.devices)

    def setup_log(self, name='train', log_dir=None, file_name=None):
        if not self.logger:
            self.logger = get_logger(
                name, log_dir, self.args.local_rank, filename=file_name)
        else:
            self.logger.warning('already exists logger')
        return self.logger

    def inject_default_parser(self):
        p = self.parser
        p.add_argument(
            '-d', '--devices', default='0',
            help='set data parallel training')
        p.add_argument(
            '-c', '--continue', type=extant_file, metavar="FILE",
            dest="continue_fpath",
            help='continue from one certain checkpoint')
        p.add_argument(
            '--local_rank', default=0, type=int,
            help='process rank on node')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, iteration):
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        self.logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}
        new_state_dict = OrderedDict()

        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict

        if self.state.optimizer:
            state_dict['optimizer'] = self.state.optimizer.state_dict()
        if self.state.scheduler:
            state_dict['scheduler'] = self.state.scheduler.state_dict()
        if self.state.iteration:
            state_dict['iteration'] = self.state.iteration

        t_io_begin = time.time()
        torch.save(state_dict, path)
        t_end = time.time()

        del state_dict
        del new_state_dict

        self.logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_io_begin - t_start, t_end - t_io_begin))

    def load_checkpoint(self, weights, is_restore=False):

        t_start = time.time()

        if weights.endswith(".pkl"):
            # for caffe2 model
            from maskrcnn_benchmark.utils.c2_model_loading import \
                load_c2_format
            loaded = load_c2_format(self.cfg, weights)
        else:
            loaded = torch.load(weights, map_location=torch.device("cpu"))

        t_io_end = time.time()
        if "model" not in loaded:
            loaded = dict(model=loaded)

        self.state.model = load_model(
            self.state.model, loaded['model'], self.logger,
            is_restore=is_restore)

        if "optimizer" in loaded:
            self.state.optimizer.load_state_dict(loaded['optimizer'])
        if "iteration" in loaded:
            self.state.iteration = loaded['iteration']
        if "scheduler" in loaded:
            self.state.scheduler.load_state_dict(loaded["scheduler"])
        del loaded

        t_end = time.time()
        self.logger.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore snapshot: {}".format(
                weights, t_io_end - t_start, t_end - t_io_end))

    def save_and_link_checkpoint(self, snapshot_dir):
        ensure_dir(snapshot_dir)
        current_iter_checkpoint = osp.join(
            snapshot_dir, 'iter-{}.pth'.format(self.state.iteration))
        self.save_checkpoint(current_iter_checkpoint)
        last_iter_checkpoint = osp.join(
            snapshot_dir, 'iter-last.pth')
        link_file(current_iter_checkpoint, last_iter_checkpoint)

    def restore_checkpoint(self, is_restore=True):
        self.load_checkpoint(self.continue_state_object, is_restore=is_restore)

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            self.logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False

    def __enter__(self):
        return self
