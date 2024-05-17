import numpy as np
from transformers import pipeline

DEFAULT_TTS_MODEL = "openai/whisper-medium.en"


class Textifier:

    def __init__(self, tts_model: str = DEFAULT_TTS_MODEL):
        """
        Initializes the Textifier object with a specified TTS model.

        Parameters:
            tts_model (str): The TTS model to be used for text-to-speech.
        """
        self.transcriber = pipeline(
            "automatic-speech-recognition",  # See [1]
            model=tts_model,
            framework="tf",
            device=None,  # Use CPU by default, set to 0 for GPU
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

    def _normalise_audio(self, raw_data):
        """
        A function to normalize audio data by maximum value.

        Parameters:
            raw_data: The raw audio data to be normalized.

        Returns:
            The normalized raw audio data.
        """
        # Prevent division by zero
        if np.max(np.abs(raw_data)) != 0:
            raw_data = raw_data.astype(np.float32)
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
        raw_data = self._normalize_audio(raw_data)
        return self._transcribe(sample_rate, raw_data)

    def to_gradio(self):
        """
        A function to convert the Textifier object to a Gradio interface compatible function.
        """
        return lambda raw_audio: self.textify(raw_audio)


# Footnotes:
# [1] https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
