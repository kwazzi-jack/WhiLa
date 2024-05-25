from pathlib import Path

import gradio
import modal
import torch
import transformers

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


@app.cls(gpu="a10g")
class Textifier:

    def __init__(self, tts_model: str = _DEFAULT_TTS_MODEL):
        """
        Initializes the Textifier object with a specified TTS model.

        Parameters:
            tts_model (str): The TTS model to be used for text-to-speech.
        """

        # Selects first GPU by default, otherwise use CPU
        device_id = 0 if torch.cuda.is_available() else None

        # Create pipeline
        self.transcriber = transformers.pipeline(
            "automatic-speech-recognition",  # See [1]
            model=tts_model,
            framework="pt",  # Using PyTorch
            device=device_id,
        )

    def _transcribe(self, sample_rate: int, raw_data: np.array) -> str:
        """
        Transcribe the raw audio data using the provided sample rate and return the transcribed text.

        Parameters:
            sample_rate (int): The sample rate of the audio data.
            raw_data (np.array): The raw audio data to transcribe.

        Returns:
            str: The transcribed text from the audio data.
        """
        try:
            result = self.transcriber({"sampling_rate": sample_rate, "raw": raw_data})
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def _normalise(self, raw_data):
        """
        A function to normalise audio data by maximum value.

        Parameters:
            raw_data: The raw audio data to be normalised.

        Returns:
            The normalised raw audio data.
        """
        raw_data = raw_data.astype(np.float32)
        if np.max(np.abs(raw_data)) != 0:
            raw_data /= np.max(np.abs(raw_data))
        return raw_data

    def textify(self, raw_audio):
        """
        Transcribe the raw audio data using the provided sample rate and return the transcribed text.

        Parameters:
            raw_audio: A tuple containing the sample rate and raw audio data.

        Returns:
            The transcribed text from the audio data.
        """
        sample_rate, raw_data = raw_audio
        raw_data = self._normalise(raw_data)
        return self._transcribe(sample_rate, raw_data)

    def to_gradio(self):
        """
        A function to convert the Textifier object to a Gradio interface compatible function.
        """
        return lambda raw_audio: self.textify(raw_audio)


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
        text = "hey brian"
        latex = self.latexifier.inference.remote(text)

        return latex

    def launch(self, share=False):
        self.interface.launch(share=share)


@app.local_entrypoint()
def main():
    GradioApp().launch(share=True)


# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
