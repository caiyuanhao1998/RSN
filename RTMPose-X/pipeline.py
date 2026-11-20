"""Detect people and estimate the 17 COCO keypoints in images and videos."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import onnxruntime as ort
import requests
import torch

from data_pipeline import (FLIP, HEIGHT, WIDTH, affine_matrix, decode_batch, fix_aspect,
                           merge_simcc_flip, oks_overlap, preprocess)
from evaluate import fuse_model, load_model

MODEL_REPO = 'CaiYuanhao/RSN'
MODEL_FILE = 'rsn-rtmpose-x.pth'
MODEL_SHA256 = '942d0be0bf9779c2bbef88d2a91cccb6ba6e122ec5bd7c12b81ff26ce958de04'
DETECTOR_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
DETECTOR_SHA256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
DETECTOR_SIZE = 640

# Colors are BGR. Consistent colors on left and right limbs avoid switching
# colors when the ordering of people changes from frame to frame.
LEFT, RIGHT, CENTER = (252, 163, 56), (92, 213, 255), (178, 222, 74)
JOINT_COLORS = [CENTER, LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT,
                LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT]
EDGES = [(0, 1, LEFT), (0, 2, RIGHT), (1, 3, LEFT), (2, 4, RIGHT),
         (5, 6, CENTER), (5, 7, LEFT), (7, 9, LEFT), (6, 8, RIGHT), (8, 10, RIGHT),
         (5, 11, LEFT), (6, 12, RIGHT), (11, 12, CENTER), (11, 13, LEFT), (13, 15, LEFT),
         (12, 14, RIGHT), (14, 16, RIGHT)]


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_detector():
    cache = Path(os.environ.get('RSN_CACHE_DIR', Path.home()/'.cache/rsn-rtmpose-x')).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/'yolox_s.onnx'
    if path.is_file() and file_sha256(path) == DETECTOR_SHA256:
        return path
    with NamedTemporaryFile(dir=cache, prefix='yolox-', suffix='.part', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with requests.get(DETECTOR_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for block in response.iter_content(chunk_size=1024*1024):
                    handle.write(block)
        if file_sha256(temporary) != DETECTOR_SHA256:
            raise ValueError('The official YOLOX-S download did not match its release checksum.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def download_pose():
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=MODEL_REPO, allow_patterns=['config.json', MODEL_FILE]))
    manifest = json.loads((snapshot / 'config.json').read_text(encoding='utf-8'))
    if MODEL_FILE not in manifest['checkpoint_files'] or manifest['checkpoint_sha256'] != MODEL_SHA256:
        raise ValueError('The Hub manifest does not match the RSN release checkpoint.')
    path = snapshot / MODEL_FILE
    if file_sha256(path) != MODEL_SHA256:
        raise ValueError('The Hub checkpoint did not match the RSN release checksum.')
    return path


class YOLOXPersonDetector:
    """YOLOX-S COCO person detections with the official ONNX grid decoding."""

    def __init__(self, path, threads=2, device='cpu'):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        if str(device).startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            device_id = torch.device(device).index or 0
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': device_id}))
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        grids, strides = [], []
        for stride in (8, 16, 32):
            grid_h, grid_w = np.indices((DETECTOR_SIZE//stride, DETECTOR_SIZE//stride))
            grid = np.stack((grid_w, grid_h), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def __call__(self, frame, confidence=.3):
        h, w = frame.shape[:2]
        scale = min(DETECTOR_SIZE/h, DETECTOR_SIZE/w)
        resized = cv2.resize(frame, (max(1, int(w*scale)), max(1, int(h*scale))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((DETECTOR_SIZE, DETECTOR_SIZE, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        image = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self.session.run(None, {self.input_name: image})[0]
        if output.shape != (1, len(self.grid), 85):
            raise ValueError('Expected the official YOLOX-S COCO ONNX output shape.')
        raw = output[0]
        scores = raw[:, 4]*raw[:, 5]
        keep = np.isfinite(scores) & (scores >= confidence)
        deltas = raw[keep, :4]
        centers = (deltas[:, :2] + self.grid[keep])*self.strides[keep]
        half_sizes = .5*np.exp(deltas[:, 2:])*self.strides[keep]
        boxes = np.concatenate((centers-half_sizes, centers+half_sizes), axis=1)/scale
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        sizes = boxes[:, 2:]-boxes[:, :2]
        good = (sizes >= 2).all(axis=1) & np.isfinite(deltas).all(axis=1)
        boxes, scores, sizes = boxes[good], scores[keep][good], sizes[good]
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        xywh = np.concatenate((boxes[:, :2], sizes), axis=1)
        picked = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, .45),
                            dtype=np.int64).ravel()
        return np.column_stack((boxes[picked], scores[picked]))


class PosePipeline:
    """Top-down pose inference on OpenCV BGR frames."""

    def __init__(self, pose_weights=None, detector_weights=None, device=None, pose_batch=16, threads=2):
        if pose_batch < 1 or threads < 1:
            raise ValueError('The pose batch and thread count must be positive.')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.pose_batch = pose_batch
        torch.set_num_threads(threads)
        self.detector = YOLOXPersonDetector(detector_weights or download_detector(), threads, self.device)
        self.model = fuse_model(load_model(pose_weights or download_pose()).eval()).to(
            self.device, memory_format=torch.channels_last)
        if self.device.type == 'cuda':
            self.model.half()

    @torch.inference_mode()
    def predict(self, frame, detection_confidence=.3, flip=False):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
            raise ValueError('Expected a nonempty BGR image with 8-bit channels.')
        if not 0 < detection_confidence < 1:
            raise ValueError('The person threshold must be between zero and one.')
        boxes = self.detector(frame, detection_confidence)
        poses = []
        for start in range(0, len(boxes), self.pose_batch):
            group = boxes[start:start+self.pose_batch]
            widths = group[:, 2:4] - group[:, :2]
            centers = group[:, :2] + .5*widths
            scales = np.stack([fix_aspect(size*1.25) for size in widths])
            images = [cv2.warpAffine(frame, affine_matrix(center, scale), (WIDTH, HEIGHT),
                                     flags=cv2.INTER_LINEAR)
                      for center, scale in zip(centers, scales)]
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
            crops = preprocess(tensor, self.device, channels_last=True)
            meta = torch.from_numpy(np.log2(np.maximum(widths, 1)/64)).to(device=self.device)
            if self.device.type == 'cuda':
                crops, meta = crops.half(), meta.half()
            extra = (meta,) if self.model.uses_proposal_size else ()
            x, y, *score_delta = self.model(crops, *extra)
            delta = score_delta[0] if score_delta else None
            if flip:
                fx, fy, *flip_delta = self.model(crops.flip(-1), *extra)
                x, y = merge_simcc_flip((x, y), (fx, fy))
                if delta is not None:
                    reflected = flip_delta[0]
                    if reflected.shape[-1] == len(FLIP):
                        reflected = reflected[:, FLIP]
                    delta = .5*(delta+reflected)
            batch = dict(center=torch.from_numpy(centers), scale=torch.from_numpy(scales),
                         bbox_score=group[:, 4], image_id=np.zeros(len(group)),
                         index=np.arange(start, start+len(group)), area=widths.prod(axis=1))
            for pose, box in zip(decode_batch((x, y), batch, score_delta=delta), group):
                if pose['score'] > 0:
                    poses.append(dict(pose, bbox=box[:4].copy(), detection_confidence=float(box[4])))
        pending = sorted(poses, key=lambda item: item['score'], reverse=True)
        kept = []
        while pending:
            kept.append(pending[0])
            pending = [pose for pose, overlap in zip(pending[1:], oks_overlap(kept[-1], pending[1:]))
                       if overlap < .9]
        return kept


def draw_poses(frame, poses, joint_confidence=.3, show_boxes=False):
    output = frame.copy()
    line_width = max(2, round(min(output.shape[:2])/320))
    joint_radius = line_width + 1
    for pose in poses:
        joints = pose['keypoints']
        visible = joints[:, 2] >= joint_confidence
        coords = np.round(joints[:, :2]).astype(np.int32)
        if show_boxes:
            x1, y1, x2, y2 = np.round(pose['bbox']).astype(np.int32)
            cv2.rectangle(output, (x1, y1), (x2, y2), CENTER, 1, cv2.LINE_AA)
        for a, b, color in EDGES:
            if visible[a] and visible[b]:
                cv2.line(output, coords[a], coords[b], (20, 23, 26), line_width+2, cv2.LINE_AA)
                cv2.line(output, coords[a], coords[b], color, line_width, cv2.LINE_AA)
        for coord, show, color in zip(coords, visible, JOINT_COLORS):
            if show:
                cv2.circle(output, coord, joint_radius+1, (20, 23, 26), -1, cv2.LINE_AA)
                cv2.circle(output, coord, joint_radius, color, -1, cv2.LINE_AA)
    return output


def process_video(pipeline, source, destination, detection_confidence=.3, joint_confidence=.3,
                  show_boxes=False, flip=False, max_side=1280, progress=None):
    """Stream decoded frames through the pipeline and write a playable H.264 MP4."""
    from imageio_ffmpeg import get_ffmpeg_exe

    destination = Path(destination)
    if destination.suffix.lower() != '.mp4':
        raise ValueError('Choose an .mp4 output for video.')
    if max_side < 2:
        raise ValueError('The video size limit must be at least two pixels.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if math.isfinite(fps) and fps > 0 else 25.
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = int(frames) if math.isfinite(frames) and frames > 0 else 0
    encoder = None
    temporary = None
    count = maximum_people = 0
    try:
        success, frame = capture.read()
        if not success:
            raise ValueError('The video has no readable frames.')
        h, w = frame.shape[:2]
        scale = min(1., max_side/max(h, w))
        width, height = max(2, round(w*scale)), max(2, round(h*scale))
        with NamedTemporaryFile(dir=destination.parent, suffix='.mp4', delete=False) as handle:
            temporary = Path(handle.name)
        command = [get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                   '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                   '-crf', '20', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        while success:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            poses = pipeline.predict(frame, detection_confidence, flip)
            rendered = draw_poses(frame, poses, joint_confidence, show_boxes)
            encoder.stdin.write(rendered.tobytes())
            maximum_people = max(maximum_people, len(poses))
            count += 1
            if progress and frames:
                progress(min(count/frames, 1.))
            success, frame = capture.read()
        encoder.stdin.close()
        encoder.stdin = None
        _, errors = encoder.communicate()
        if encoder.returncode:
            raise RuntimeError(f'Video encoding failed. {errors.decode(errors="replace").strip()}')
        temporary.replace(destination)
    finally:
        capture.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.communicate()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(frames=count, max_people=maximum_people, fps=fps, output=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pose-weights', type=Path)
    parser.add_argument('--detector-weights', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--pose-batch', type=int, default=16)
    parser.add_argument('--detection-confidence', type=float, default=.3)
    parser.add_argument('--joint-confidence', type=float, default=.3)
    parser.add_argument('--show-boxes', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--max-side', type=int, default=1280)
    args = parser.parse_args()
    pipeline = PosePipeline(args.pose_weights, args.detector_weights, args.device, args.pose_batch)
    image = cv2.imread(str(args.input))
    if image is not None:
        people = pipeline.predict(image, args.detection_confidence, args.flip)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), draw_poses(image, people, args.joint_confidence, args.show_boxes)):
            raise OSError(f'Could not write {args.output}')
        report = dict(people=len(people), output=str(args.output))
    else:
        report = process_video(pipeline, args.input, args.output, args.detection_confidence,
                               args.joint_confidence, args.show_boxes, args.flip, args.max_side)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
