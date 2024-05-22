import os
from pathlib import Path

import modal
from modal import App, Image

from whila.textify import Textifier

_DEFAULT_MODEL_DIR = Path("models")


def download_model_to_image(
    model_name: str, model_dir: str | Path = _DEFAULT_MODEL_DIR
) -> None:
    """Based on [1] from modal.com"""
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    # Account for strings
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    # Make directories
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download via huggingface
    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )

    # Move models
    move_cache()


def _init_image(
    model_dir, model_names, python_version: str = "3.10", timeout: int = 1200
) -> Image:

    # Create base image template with required modules
    image = (
        Image.debian_slim(python_version=python_version)
        .pip_install(
            "vllm==0.4.2",
            "torch==2.3.0",
            "transformers==4.40.2",
            "ray==2.10.0",
            "hf-transfer==0.1.6",
            "huggingface_hub==0.22.2",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

    # Download models
    for model_name in model_names:
        image.run_function(
            download_model_to_image,
            timeout=timeout,
            kwargs={"model_dir": model_dir, "model_name": model_name},
            secrets=[modal.Secret.from_name("huggingface-secret")],
        )


def _init_app(image, tts_model) -> App: ...


def init_modal(tts_model: str = "openai/whisper-medium.en", gpu=None):
    app = App()


# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
