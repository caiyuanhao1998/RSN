"""Gradio interface for RSN image and video inference."""
from functools import lru_cache, partial
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import gradio as gr
from gradio.utils import get_upload_folder
import torch

from pipeline import MODEL_REPO, PosePipeline, draw_poses, process_video

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
TITLE = "RSN · ECCV 2020"
CODE = "https://github.com/caiyuanhao1998/RSN"
WEIGHTS = f"https://huggingface.co/{MODEL_REPO}"
SITE = "https://caiyuanhao1998.github.io/RSN-Project/"
PAPER = "https://arxiv.org/abs/2003.04030"


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


def media_examples(sources):
    examples = []
    with gr.Row(elem_classes="pose-samples"):
        gr.HTML('<span>Examples</span>', apply_default_css=False, elem_classes="pose-samples-label")
        for path in sources:
            thumbnail = path.with_suffix(".jpg")
            button = gr.Button(
                path.stem.title(), icon=thumbnail if thumbnail.is_file() else None,
                size="sm", scale=0, min_width=0, elem_classes="pose-example",
            )
            examples.append((button, path))
    return examples


def media_controls(kind):
    with gr.Column(min_width=0, elem_id=f"{kind}-controls", elem_classes="pose-mode-controls"):
        with gr.Row(elem_classes="pose-actions"):
            upload = gr.UploadButton(
                f"Upload {kind}", file_types=[kind], size="md", min_width=0,
                elem_classes="pose-upload",
            )
            run = gr.Button("Estimate poses", variant="primary", size="md", min_width=0)
        status = gr.Markdown(elem_classes="pose-status")
    return upload, run, status


def set_source(path):
    return str(path), None, ""


def clear_result():
    return None, ""


def connect_media(fn, source, settings, result, controls, examples):
    upload, run, status = controls
    inputs = [source, *settings]
    outputs = [result, status]
    immediate = dict(queue=False, show_progress="hidden", api_visibility="private")
    upload.upload(set_source, upload, [source, *outputs], **immediate)
    gr.on([source.input, source.clear], clear_result, outputs=outputs, **immediate)
    run.click(fn, inputs, outputs, api_name=fn.__name__, concurrency_id="pose")
    for button, path in examples:
        button.click(partial(set_source, path), outputs=[source, *outputs], **immediate)


def build_demo():
    panel = dict(show_label=False, min_width=0, height="var(--pose-media-height)",
                 elem_classes="pose-media")
    with gr.Blocks(title=TITLE, fill_width=True, delete_cache=(3600, 86400)) as demo:
        gr.HTML(
            f'<header id="pose-heading"><h1>{TITLE}</h1><div class="pose-meta">'
            '<p>Multi-person pose estimation</p><nav aria-label="Project resources">'
            f'<a href="{SITE}" target="_blank" rel="noopener">Project page</a>'
            f'<a href="{PAPER}" target="_blank" rel="noopener">Paper</a>'
            f'<a href="{CODE}" target="_blank" rel="noopener">GitHub</a>'
            f'<a href="{WEIGHTS}" target="_blank" rel="noopener">Weights</a></nav></div></header>',
            apply_default_css=False, elem_id="pose-header",
        )
        with gr.Tabs(elem_id="pose-tabs"):
            with gr.Tab("Image", elem_id="image-tab"):
                image_examples = media_examples(sorted(EXAMPLES.glob("*.jpg")))
                with gr.Row(elem_classes="pose-panels"):
                    with gr.Column(min_width=0, elem_classes="pose-pane"):
                        gr.HTML("<h2>Original image</h2>", apply_default_css=False)
                        input_image = gr.Image(
                            value=str(EXAMPLES / "basketball.jpg"), label="Original image",
                            type="numpy", image_mode="RGB", interactive=True,
                            sources=["upload", "clipboard", "webcam"], buttons=["fullscreen"],
                            elem_id="image-source", **panel,
                        )
                    with gr.Column(min_width=0, elem_classes="pose-pane"):
                        gr.HTML("<h2>Estimated poses</h2>", apply_default_css=False)
                        output_image = gr.Image(
                            label="Estimated poses",
                            interactive=False, buttons=["fullscreen", "download"],
                            elem_id="image-result", **panel,
                        )
            with gr.Tab("Video", elem_id="video-tab"):
                video_sources = sorted((EXAMPLES / "videos").glob("*.mp4"))
                video_examples = media_examples(video_sources)
                with gr.Row(elem_classes="pose-panels"):
                    with gr.Column(min_width=0, elem_classes="pose-pane"):
                        gr.HTML("<h2>Original video</h2>", apply_default_css=False)
                        input_video = gr.Video(
                            value=video_sources[0] if video_sources else None,
                            label="Original video",
                            sources=["upload", "webcam"], interactive=True,
                            elem_id="video-source", **panel,
                        )
                    with gr.Column(min_width=0, elem_classes="pose-pane"):
                        gr.HTML("<h2>Estimated poses</h2>", apply_default_css=False)
                        output_video = gr.Video(
                            label="Estimated poses",
                            format="mp4", interactive=False, buttons=["download"],
                            elem_id="video-result", **panel,
                        )
        with gr.Column(min_width=0, elem_id="pose-controls"):
            image_controls = media_controls("image")
            video_controls = media_controls("video")
            with gr.Accordion("Inference Settings", open=True, elem_id="pose-settings"):
                with gr.Row():
                    person = gr.Slider(.05, .95, value=.30, step=.01, label="Person confidence")
                    joint = gr.Slider(.05, .95, value=.30, step=.01, label="Joint confidence")
                    resolution = gr.Dropdown(
                        [640, 960, 1280], value=960, label="Video output long side (px)",
                        elem_id="video-resolution",
                    )
                with gr.Row():
                    boxes = gr.Checkbox(label="Show person boxes", scale=0, min_width=160)
                    flip = gr.Checkbox(label="Use flip inference", scale=0, min_width=160)
        options = [person, joint, boxes, flip]
        connect_media(estimate_image, input_image, options, output_image, image_controls, image_examples)
        connect_media(estimate_video, input_video, [*options, resolution], output_video, video_controls, video_examples)
    return demo.queue(default_concurrency_limit=1)


demo = build_demo()

if __name__ == "__main__":
    theme = gr.themes.Base(primary_hue="zinc", secondary_hue="zinc", neutral_hue="zinc").set(
        body_background_fill="white", body_text_color_subdued="#626267",
        body_text_color_subdued_dark="#bdbdc2",
        button_primary_background_fill="#262626", button_primary_background_fill_hover="#111111",
        button_primary_text_color="white", button_primary_border_color="#262626",
        button_primary_background_fill_dark="#ededed", button_primary_background_fill_hover_dark="white",
        button_primary_text_color_dark="#191919", button_primary_border_color_dark="#ededed",
    )
    demo.launch(theme=theme, css_paths=ROOT / "app.css", footer_links=["api", "settings"])
