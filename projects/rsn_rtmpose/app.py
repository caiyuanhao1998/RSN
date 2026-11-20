"""Gradio interface for RSN + RTMPose-X image and video inference."""
from functools import lru_cache
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import gradio as gr
from gradio.utils import get_upload_folder
import torch

from pipeline import PosePipeline, draw_poses, process_video

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
CODE = "https://github.com/caiyuanhao1998/RSN"
WEIGHTS = "https://huggingface.co/CaiYuanhao/RSN-RTMPose-X"
SITE = "https://caiyuanhao1998.github.io/RSN-Project/"
PAPER = "https://arxiv.org/abs/2003.04030"

CSS = """
.gradio-container {max-width: 1360px !important; margin: 0 auto !important;}
#pose-heading {margin: 6px 0 16px; padding: 26px 30px; border-radius: 16px;
  background: #0d202c; color: #d9edf2; border: 1px solid #284653;}
#pose-heading h1 {margin: 0 0 8px; color: #f7fcff; font-size: clamp(26px, 4vw, 40px);
  letter-spacing: -.035em; line-height: 1.12;}
#pose-heading p {margin: 0 0 14px; line-height: 1.55;}
#pose-heading nav {display: flex; flex-wrap: wrap; gap: 7px 18px;}
#pose-heading a {color: #82eadb; font-weight: 600; text-decoration: none;}
#pose-heading a:hover, #pose-heading a:focus-visible {text-decoration: underline;}
#pose-controls {border-radius: 12px; padding: 12px; background: var(--background-fill-secondary);}
#pose-controls h2 {margin-top: 0; font-size: 1rem;}
.pose-status {min-height: 1.5rem; margin-top: 4px;}
@media (max-width: 700px) {#pose-heading {padding: 20px; margin-bottom: 8px;}}
"""


@lru_cache(maxsize=1)
def get_pipeline():
    device = os.environ.get("RSN_DEVICE")
    cuda = torch.cuda.is_available() if not device else device.startswith("cuda")
    return PosePipeline(
        pose_weights=os.environ.get("RSN_POSE_WEIGHTS"),
        detector_weights=os.environ.get("RSN_DETECTOR_WEIGHTS"),
        device=device,
        threads=int(os.environ.get("RSN_THREADS", "2")),
        pose_batch=16 if cuda else 4,
    )


def estimate_image(image, person_score, joint_score, show_boxes, flip):
    if image is None:
        raise gr.Error("Upload an image or select an example.")
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    poses = get_pipeline().predict(frame, person_score, flip)
    result = draw_poses(frame, poses, joint_score, show_boxes)
    count = len(poses)
    message = "1 person detected." if count == 1 else f"{count} people detected."
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), message


def estimate_video(video, person_score, joint_score, show_boxes, flip, max_side, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video or select an example.")
    progress(0, desc="Loading pose estimator")
    pipeline = get_pipeline()
    cache = Path(get_upload_folder())
    cache.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache, suffix=".mp4", prefix="rsn-pose-", delete=False) as handle:
        output = Path(handle.name)

    def report(fraction):
        progress(fraction, desc="Estimating poses")

    try:
        stats = process_video(pipeline, video, output, person_score, joint_score,
                              show_boxes, flip, int(max_side), progress=report)
    except (ValueError, OSError, RuntimeError) as exc:
        output.unlink(missing_ok=True)
        raise gr.Error(str(exc)) from exc
    return str(output), f"{stats['frames']} frames processed. Peak detections per frame {stats['max_people']}."


def build_demo():
    with gr.Blocks(title="RSN + RTMPose-X", delete_cache=(3600, 86400)) as demo:
        gr.HTML(
            '<header id="pose-heading"><h1>RSN + RTMPose-X</h1>'
            '<p>Multi-person pose estimation for images and video.</p>'
            f'<nav aria-label="Project links"><a href="{PAPER}">RSN paper</a>'
            f'<a href="{CODE}">GitHub</a><a href="{WEIGHTS}">Weights</a>'
            f'<a href="{SITE}">Project page</a></nav></header>'
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=218, elem_id="pose-controls"):
                gr.Markdown("## Settings")
                person = gr.Slider(.05, .95, value=.30, step=.01, label="Person confidence")
                joint = gr.Slider(.05, .95, value=.30, step=.01, label="Joint confidence")
                with gr.Accordion("More options", open=False):
                    boxes = gr.Checkbox(label="Show person boxes")
                    flip = gr.Checkbox(label="Use flip inference", info="Runs a second pose forward pass")
                    resolution = gr.Dropdown([640, 960, 1280], value=960, label="Video output long side (px)")
            with gr.Column(scale=4, min_width=300):
                with gr.Tabs():
                    with gr.Tab("Image"):
                        with gr.Row():
                            input_image = gr.Image(label="Input", type="numpy", image_mode="RGB",
                                                   sources=["upload", "clipboard", "webcam"], height=360)
                            output_image = gr.Image(label="Poses", interactive=False, height=360)
                        run_image = gr.Button("Estimate image poses", variant="primary")
                        image_status = gr.Markdown(elem_classes=["pose-status"])
                        examples = [[str(file)] for file in sorted(EXAMPLES.glob("*.jpg"))]
                        if examples:
                            gr.Examples(examples, inputs=[input_image], cache_examples=False)
                    with gr.Tab("Video"):
                        with gr.Row():
                            input_video = gr.Video(label="Input", sources=["upload", "webcam"], height=360)
                            output_video = gr.Video(label="Poses", format="mp4", interactive=False, height=360)
                        run_video = gr.Button("Estimate video poses", variant="primary")
                        video_status = gr.Markdown(elem_classes=["pose-status"])
                        examples = [[str(file)] for file in sorted(EXAMPLES.glob("*.mp4"))]
                        if examples:
                            gr.Examples(examples, inputs=[input_video], cache_examples=False)
        run_image.click(estimate_image, inputs=[input_image, person, joint, boxes, flip],
                        outputs=[output_image, image_status], api_name="estimate_image", concurrency_id="pose")
        run_video.click(estimate_video, inputs=[input_video, person, joint, boxes, flip, resolution],
                        outputs=[output_video, video_status], api_name="estimate_video", concurrency_id="pose")
    return demo.queue(default_concurrency_limit=1)


demo = build_demo()

if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate)
    demo.launch(theme=theme, css=CSS)
