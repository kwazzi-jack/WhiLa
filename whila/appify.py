from pathlib import Path

import gradio
import modal

from whila.textify import Textifier

_DEFAULT_VOLUME = {"/whila": modal.Volume.from_name("whila-volume")}
_DEFAULT_MODEL_DIR = Path("/whila/models")
_DEFAULT_TTS_MODEL = "openai/whisper-medium.en"
_PIP_PACKAGES = ["requests"]
_DEFAULT_PYVERSION = "3.10"
_DEFAULT_IMAGE = modal.Image.debian_slim(python_version=_DEFAULT_PYVERSION)
_DEFAULT_GPU_CONFIG = modal.gpu.L4()


def init_modal_instance():
    return modal.App(
        "whila-app",
        image=_DEFAULT_IMAGE.pip_install(*_PIP_PACKAGES),
        volumes=_DEFAULT_VOLUME,
    )


app = init_modal_instance()


# refactor this into a separate file, modalize
@app.cls(gpu="a10g")
class Latexifier:  # use vllm probably
    @modal.enter()
    def load_model(self):
        print("entering")
        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L108-L111
        pass

    @modal.method()
    def inference(self, text):
        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L118-L131
        print(text)
        return text


class GradioApp:

    def __init__(self):
        self.textifier = Textifier()
        self.latexifier = modal.Cls.lookup("whila-app", "Latexifer")
        self._create_gradio_interface()

    def _create_gradio_interface(self):
        self.interface = gradio.Interface(
            self._audio_to_latex(),
            gradio.Audio(sources=["microphone"]),
            "text",
        )

    def _audio_to_latex(self, audio):
        # text = self.textifer(audio)
        text
        latex = self.latexifier.inference.remote(text)

        return latex

    def launch(self, share=False):
        self.interface.launch(share=share)


if __name__ == "__main__":
    GradioApp().launch(share=True)
# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
