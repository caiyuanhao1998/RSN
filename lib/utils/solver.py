"""
@author: Yuanhao Cai
@date:  2020.03
"""

import torch.optim as optim

def make_optimizer(cfg, model, num_gpu):
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                lr=cfg.SOLVER.BASE_LR * num_gpu,
                betas=(0.9, 0.999), eps=1e-08,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    w_iters = cfg.SOLVER.WARMUP_ITERS
    w_fac = cfg.SOLVER.WARMUP_FACTOR
    max_iter = cfg.SOLVER.MAX_ITER
    lr_lambda = lambda iteration : w_fac + (1 - w_fac) * iteration / w_iters \
            if iteration < w_iters \
            else 1 - (iteration - w_iters) / (max_iter - w_iters)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    return scheduler

