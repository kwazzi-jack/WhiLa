import numpy as np
import torch
from transformers import pipeline

DEFAULT_TTS_MODEL = "openai/whisper-medium.en"


class Textifier:

    def __init__(self, tts_model: str = DEFAULT_TTS_MODEL):
        """
        Initializes the Textifier object with a specified TTS model.

        Parameters:
            tts_model (str): The TTS model to be used for text-to-speech.
        """

        # Selects first GPU by default, otherwise use CPU
        device_id = 0 if torch.cuda.is_available() else None

        # Create pipeline
        self.transcriber = pipeline(
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


# Footnotes:
# [1] https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
