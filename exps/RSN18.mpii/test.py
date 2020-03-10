"""
@author: Yuanhao Cai
@date:  2020.03
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.distributed as dist

from cvpack.utils.logger import get_logger

from config import cfg
from network import RSN
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process, synchronize, all_gather
from lib.utils.transforms import flip_back


def get_results(outputs, centers, scales, kernel=11, shifts=[0.25]):
    scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
            cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                    int(round(x) + 1e-9)]
        # aligned or not ...
        kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
                centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
                centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores 
    
    return preds, maxvals


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results = list() 
    cpu_device = torch.device("cpu")

    data = tqdm(data_loader) if is_main_process() else data_loader
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch

        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.to(cpu_device).numpy()

            if cfg.TEST.FLIP:
                imgs_flipped = np.flip(imgs.to(cpu_device).numpy(), 3).copy()
                imgs_flipped = torch.from_numpy(imgs_flipped).to(device)
                outputs_flipped = model(imgs_flipped)
                outputs_flipped = outputs_flipped.to(cpu_device).numpy()
                outputs_flipped = flip_back(
                        outputs_flipped, cfg.DATASET.KEYPOINT.FLIP_PAIRS)
                
                outputs = (outputs + outputs_flipped) * 0.5

        centers = np.array(centers)
        scales = np.array(scales)
        preds, maxvals = get_results(outputs, centers, scales,
                cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)

        preds = np.concatenate((preds, maxvals), axis=2)
        results.append(preds)

    return results 


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, logger):
    if is_main_process():
        logger.info("Accumulating ...")
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for pred in all_predictions:
        predictions.extend(pred)
    predictions = np.vstack(predictions) 
    
    return predictions


def inference(model, data_loader, logger, device="cuda"):
    predictions = compute_on_dataset(model, data_loader, device)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(
            predictions, logger)

    if not is_main_process():
        return

    return predictions    
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    args = parser.parse_args()

    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if is_main_process() and not os.path.exists(cfg.TEST_DIR):
        os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.")

    model = RSN(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(cfg.MODEL.DEVICE)

    model_file = os.path.join(cfg.OUTPUT_DIR, "iter-{}.pth".format(args.iter))
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    data_loader = get_test_loader(cfg, num_gpus, args.local_rank, 'val',
            is_dist=distributed)

    results = inference(model, data_loader, logger, device)
    synchronize()

    if is_main_process():
        logger.info("Get all results.")

        data_loader.ori_dataset.evaluate(results)


if __name__ == '__main__':
    main()
