from pathlib import Path

import gradio
import modal
from modal import App, Image, Volume

from whila.textify import Textifier

_DEFAULT_VOLUME = {"/whila": Volume.from_name("whila-volume")}
_DEFAULT_MODEL_DIR = Path("/whila/models")
_DEFAULT_TTS_MODEL = "openai/whisper-medium.en"
_PIP_PACKAGES = [
    "git+https://github.com/kwazzi-jack/WhiLa.git",
    "torch",
    "transformers",
]
_DEFAULT_PYVERSION = "3.10"
_DEFAULT_IMAGE = Image.debian_slim(python_version=_DEFAULT_PYVERSION)
_DEFAULT_GPU_CONFIG = modal.gpu.L4()


def init_modal_instance():
    return App(
        "whila-app",
        image=_DEFAULT_IMAGE.pip_install(*_PIP_PACKAGES),
        volumes=_DEFAULT_VOLUME,
        gpu_config=_DEFAULT_GPU_CONFIG,
    )


class ModalApp:

    def __init__(self):
        self.textifier = Textifier()
        self._create_gradio_interface()

    def _create_gradio_interface(self):
        self.interface = gradio.Interface(
            self.textifier.to_gradio(),
            gradio.Audio(sources=["microphone"]),
            "text",
        )

    def launch(self, live=False):
        self.interface.launch(live=live)


# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
