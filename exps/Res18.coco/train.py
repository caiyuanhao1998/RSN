"""
@author: Yuanhao Cai
@date:  2020.03
"""

import argparse
import time

import torch
from tensorboardX import SummaryWriter

from cvpack.torch_modeling.engine.engine import Engine
from cvpack.utils.pyt_utils import ensure_dir

from config import cfg
from network import RSN 
from lib.utils.dataloader import get_train_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer


def main():
    parser = argparse.ArgumentParser()

    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name='log.txt')
        args = parser.parse_args()
        ensure_dir(cfg.OUTPUT_DIR)

        model = RSN(cfg, run_efficient=cfg.RUN_EFFICIENT)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        num_gpu = len(engine.devices) 
        # default num_gpu: 8, adjust iter settings
        cfg.SOLVER.CHECKPOINT_PERIOD = \
                int(cfg.SOLVER.CHECKPOINT_PERIOD * 8 / num_gpu)
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * 8 / num_gpu)
        optimizer = make_optimizer(cfg, model, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                broadcast_buffers=False, )

        if engine.continue_state_object:
            engine.restore_checkpoint(is_restore=False)
        else:
            if cfg.MODEL.WEIGHT:
                engine.load_checkpoint(cfg.MODEL.WEIGHT, is_restore=False)

        data_loader = get_train_loader(cfg, num_gpu=num_gpu, is_dist=engine.distributed)

        # ------------ do training ---------------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(
            torch.__version__))

        max_iter = len(data_loader)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)

        model.train()

        time1 = time.time()
        for iteration, (images, valids, labels) in enumerate(
                data_loader, engine.state.iteration):
            iteration = iteration + 1
            images = images.to(device)
            valids = valids.to(device)
            labels = labels.to(device)

            scheduler.step()
            loss_dict = model(images, valids, labels)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if cfg.RUN_EFFICIENT:
                del images, valids, labels, losses

            if engine.local_rank == 0:
                if iteration % 20 == 0 or iteration == max_iter:
                    log_str = 'Iter:%d, LR:%.1e, ' % (
                        iteration, optimizer.param_groups[0]["lr"] / num_gpu)
                    for key in loss_dict:
                        tb_writer.add_scalar(
                            key,  loss_dict[key].mean(), global_step=iteration)
                        log_str += key + ': %.3f, ' % float(loss_dict[key])

                    time2 = time.time()
                    elapsed_time = time2 - time1
                    time1 = time2
                    required_time = elapsed_time / 20 * (max_iter - iteration)
                    hours = required_time // 3600
                    mins = required_time % 3600 // 60
                    log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

                    logger.info(log_str)

            if iteration % checkpoint_period == 0 or iteration == max_iter:
                engine.update_iteration(iteration)
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)

            if iteration >= max_iter:
                logger.info('Finish training process!')
                break


if __name__ == "__main__":
    main()
