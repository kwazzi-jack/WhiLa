import os
import re
import time
from pathlib import Path

import gradio
import modal


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


VOLUME = {"/whila": modal.Volume.from_name("whila-volume")}
MODEL_DIR = Path("/model")
TTS_MODEL = "openai/whisper-medium.en"
TTL_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
APT_PACKAGES = ["git"]
PIP_PACKAGES = [
    "vllm==0.4.0.post1",
    "torch==2.1.2",
    "torchaudio==2.1.2",
    "transformers==4.39.3",
    "ray==2.10.0",
    "hf-transfer==0.1.6",
    "huggingface_hub==0.22.2",
]
OPT_PIP_PACKAGES = ["gradio", "librosa"]

PY_VERSION = "3.10"
IMAGE = (
    modal.Image.debian_slim(python_version=PY_VERSION)
    .pip_install(*PIP_PACKAGES)
    .pip_install(*OPT_PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": TTL_MODEL},
        secrets=[modal.Secret.from_name("Modal-to-HF-secret")],
    )
)
GPU_CONFIG = modal.gpu.A100(count=2)

EXTRACT_TEMPLATE = r"""
**Extraction Task:**

1. **Commands:**
   - The input must begin with a command from the following list:
     
     commands = [equation, align, next, end]
     
   - The command must be at the start of the input text and must match exactly.

2. **Values:**
   - The value follows the command and represents a mathematical expression, theorem, or result.
   - Capture the value as it is, with minimal editing, and focus on the mathematical object being described.
   - The only editing allowed is to remove conversational aspects not related to the description.
   - If a value cannot be identified, return "invalid-value".

3. **Input and Output:**
   - Input will be provided in the format:
     
     ?INPUT=<input_text>?
     
   - Output should be structured strictly as:
     
     ?COMMAND=<command_identified>, VALUE=<value_identified>?
     
   - If an invalid command or value is found, return the error in the format:
     
     ?ERROR=<invalid_error>?
     

4. **Strict Output Example:**
   - Example Input:
     
    ?INPUT=equation cosine brackets of pi divided by 2 plus 2 theta close brackets?
     

   - Example Output:
     
    ?COMMAND=equation, VALUE=cosine brackets of pi divided by 2 plus 2 theta close brackets?
     

5. **Instructions:**
   - The output should be only the extracted command and value in the specified format.
   - Do not provide explanations or additional text.
   - Adhere strictly to the example format for every valid input.
   - If the input is invalid, only return the error message as specified.

Given the above, here is the input:

?INPUT={input_text}?

The output is strictly as:

"""


CONVERT_TEMPLATE = r"""
**Conversion Task:**

1. **Input:**
   - The input will be provided in the format:
     
     ?VALUE=<value_text>?
     

2. **Output:**
   - Convert the given `VALUE` text into valid LaTeX code that can be used in math-mode.
   - Focus on the core mathematical expressions without unnecessary brackets or symbols unless they are mathematically required.
   - The LaTeX code should be formatted nicely and use normal conventions for writing mathematics.
   - The output should be structured strictly as:
     
     ?OUTPUT=<output_latex>?
     

3. **Strict Output Example:**
   - Example Input:
     
     ?VALUE=sum of two to the power of eye calculated from eye equals one to big n?
     

   - Example Output:
     
     ?OUTPUT=\sum\limits^N_{{i = 1}} 2^i?
     
   - Example Input:
     
     ?VALUE=cosine brackets of pie divided by two plus two theta close brackets?
     
   - Example Output:
     
     ?OUTPUT=\cos\left(\frac{{\pi}}{{2}} + 2\theta\right)?
     

4. **Instructions:**
   - The output should be only the converted LaTeX code in the specified format.
   - Do not provide explanations or additional text.
   - Adhere strictly to the example format for every valid input.
   - If the input is invalid, only return the error message as specified.

Given the above, here is the input:

?VALUE={value_text}?

The output is:

"""


# Define the regex pattern with named groups
pattern = (
    r"\?COMMAND=(?P<COMMAND>[^,]+),\s*VALUE=(?P<VALUE>[^\?]+)\?"  # Command and Value together
    r"|\?ERROR=(?P<ERROR>[^\?]+)\?"  # Alternatively, Error
    r"|\?OUTPUT=(?P<OUTPUT>[^\?]+)\?"  # Alternatively, Output
)

app = modal.App(
    "whila-app",
    image=IMAGE,
    volumes=VOLUME,
)


# refactor this into a separate file, modalize
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    secrets=[modal.Secret.from_name("Modal-to-HF-secret")],
    image=IMAGE,
)
class Latexifier:  # use vllm probably
    extract_template = EXTRACT_TEMPLATE
    convert_template = CONVERT_TEMPLATE
    search_pattern = pattern

    @modal.enter()
    def load_model(self):
        import torch
        import vllm

        print("Running load_model...")

        self.model_params = vllm.SamplingParams(
            temperature=0.0,
            top_p=0.005,
            top_k=64,
            max_tokens=50,  # Adjusted for deterministic output
        )
        # Create TTL LLM agent
        self.ttl = vllm.LLM(
            MODEL_DIR,
            dtype=torch.bfloat16,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.9,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            # skip_special_tokens=True,
        )

        # Save point for `align`
        self.save = ""

        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L108-L111

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()

    def _search_over_response(self, response):

        # Apply the regex to the input text
        match = re.search(pattern, response)

        # Extract the matched groups into a dictionary
        if match:
            return {
                key: value
                for key, value in match.groupdict().items()
                if value is not None
            }
        else:
            return {}

    def extractor(self, text):
        """Format extractor text to get command and value"""
        return self.extract_template.format(input_text=text)

    def converter(self, value):
        """Format converter text to for converting a value to latex"""
        return self.convert_template.format(value_text=value)

    def _align_symbols(latex, mode="="):
        """Adds alignment characters to an align environment."""

        match mode.lower():
            case "=":
                return latex.replace(r"=", r"&=")
            case "left" | "l":
                return latex.replace(r"\t", r"\t&")
            case "right" | "r":
                return latex.replace(r"\end", r"&\end").replace(r"\n", r"&\n")

    def cleanup(self, exec_type, output):
        """Clean-up the output from the LLM based on the execution
        type."""

        outputs = self._search_over_response(output)
        print("DICT:", outputs)

        # If an error was produced, raise it
        if "ERROR" in outputs:
            error = self.outputs["ERROR"].capitalize()
            raise ValueError(f"Error from extractor: '{error}'")

        # Check if it contains a command or a value field
        if (
            exec_type == "extract"
            and "COMMAND" not in outputs
            and "VALUE" not in outputs
        ):
            raise ValueError(f"Cannot find command or value in '{outputs}'")

        # Check if it contains a output field
        if exec_type == "convert" and "OUTPUT" not in outputs:
            raise ValueError(f"Cannot find output in '{outputs}'")

        # Output is from extractor, return command and value
        if exec_type == "extract":
            command = outputs["COMMAND"].strip().lower()
            value = outputs["VALUE"].strip()
            return command, value

        # Output is from converter, return latex
        elif exec_type == "convert":
            return outputs["OUTPUT"].strip()

        # Unknown executor type
        else:
            raise ValueError(f"Unknown executor type `{exec_type}`")

    def extract(self, text):
        """Perform extraction of the command and value from the LLM based on
        input audio."""
        extractor = self.extractor(text)
        response = self.ttl.generate(extractor, sampling_params=self.model_params)
        extracted_text = response[0].outputs[0].text

        print("EXTRACTED:", extracted_text)
        return self.cleanup("extract", extracted_text)

    def convert(self, value):
        """Perform conversion of the value from the extractor using the LLM
        to get the corresponding latex."""
        converter = self.converter(value)
        response = self.ttl.generate(converter, sampling_params=self.model_params)
        converted_text = response[0].outputs[0].text

        print("CONVERTED:", converted_text)
        return self.cleanup("convert", converted_text)

    def finalize(self, command, latex):
        """Based on the command, identify and format how the latex should be wrapped"""
        match command.lower():

            # Equation environment
            case "equation":
                return r"$$\begin{{equation}}{contents}\end{{equation}}$$".format(
                    contents=latex
                )

            # Error: Unknown environment
            case _:
                raise ValueError(f"Unknown command type '{command}'")

    @modal.method()
    def latexify(self, text):
        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L118-L131
        print("Running latexify...")
        try:
            command, value = self.extract(text)

            print("COMMAND:", command)
            print("VALUE:", value)

            latex = self.convert(value)

            print("LATEX:", latex)

            output = self.finalize(command, latex)
        except Exception as e:
            output = e
        return output


@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    secrets=[modal.Secret.from_name("Modal-to-HF-secret")],
    image=IMAGE,
)
class Textifier:

    @modal.enter()
    def load_model(self):
        """
        Initializes the Textifier object with a specified TTS model.

        Parameters:
            tts_model (str): The TTS model to be used for text-to-speech.
        """

        import torch
        import transformers

        # Selects first GPU by default, otherwise use CPU
        device_id = 0 if torch.cuda.is_available() else None

        # Create pipeline
        self.transcriber = transformers.pipeline(
            "automatic-speech-recognition",  # See [1]
            model=TTS_MODEL,
            framework="pt",  # Using PyTorch
            device=device_id,
        )

    def _transcribe(self, sample_rate, raw_data) -> str:
        """
        Transcribe the raw audio data using the provided sample rate and return the transcribed text.

        Parameters:
            sample_rate (int): The sample rate of the audio data.
            raw_data (np.array): The raw audio data to transcribe.

        Returns:
            str: The transcribed text from the audio data.
        """
        import librosa

        try:
            # Ensure the audio is mono
            if len(raw_data.shape) > 1 and raw_data.shape[1] > 1:
                raw_data = librosa.to_mono(raw_data.T)

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

        import numpy as np

        raw_data = raw_data.astype(np.float32)
        if np.max(np.abs(raw_data)) != 0:
            raw_data /= np.max(np.abs(raw_data))
        return raw_data

    @modal.method()
    def textify(self, raw_audio):
        """
        Transcribe the raw audio data using the provided sample rate and return the transcribed text.

        Parameters:
            raw_audio: A tuple containing the sample rate and raw audio data.

        Returns:
            The transcribed text from the audio data.
        """
        print("Running textify...")
        if raw_audio:
            sample_rate, raw_data = raw_audio
        else:
            return ""

        raw_data = self._normalise(raw_data)
        output = self._transcribe(sample_rate, raw_data)
        print("CAPTURED:", output)
        return output


from fastapi import FastAPI

web_app = FastAPI()


@app.function(
    image=IMAGE,
    volumes=VOLUME,
    gpu=GPU_CONFIG,
    concurrency_limit=1,
    container_idle_timeout=600,
)
@modal.asgi_app()
def main():
    textifier = modal.Cls.lookup("whila-app", "Textifier")
    latexifier = modal.Cls.lookup("whila-app", "Latexifier")

    def whila_function(raw_audio):
        result = "*Textifier*:\n{text}\n\n*Latexifier*:\n{latex}"
        text = textifier.textify.remote(raw_audio)
        if text:
            latex = latexifier.latexify.remote(text)
        else:
            latex = "Error with textifier, no output"

        return result.format(text=text, latex=latex)

    # Define the Gradio interface
    interface = gradio.Interface(
        fn=whila_function,
        inputs=gradio.Audio(sources=["microphone"]),
        outputs="markdown",
        live=True,
    )

    # Create a block for combining interfaces
    with gradio.Blocks() as demo:

        gradio.Markdown("# WhiLa Demo - Backdrop Build V4")
        gradio.Markdown(
            """
            ## Introduction
            Welcome to the WhiLa demo! This application showcases the integration of speech-to-text technology using OpenAI's Whisper and LaTeX conversion using Meta's Llama3-8B.
            
            ### How it Works
            WhiLa stands for *Whisper-to-LaTeX* and its goal is to convert spoken mathematical expressions into valid LaTeX code. The framework consists of two components:
            
            - **Textifier**: This is the *Speech-To-Text* (STT) layer that converts the audio input to spoken word. This uses the OpenAI Whisper model to achieve.

            - **Latexifier**: This is the *Text-To-LaTeX* (TTL) layer that converts the structured input text into valid LaTeX code based on the recorded input from the Textifier. This uses Meta's Llama3-8B-Instruct model with specific structured prompts to achieve.

            - **Modal**: The current demo is running on a [modal.com](https://modal.com) container and allows for the processing of WhiLa. Big thank you to Modal and **Charles Frye** at Modal for your assistance towards getting this project running on the platform!
            
            ### How to Use
            1. Click the record button to start recording.
            2. Begin your recording with the word **equation**.
            3. Describe your mathematical expression verbally.
            4. Wait for the output.

            ### Tips & Notes
            - WhiLa is designed to minimize assumptions or edits to your description. If your input is ambiguous, WhiLa may interpret it creatively!
            - Named mathematical concepts or theorems can be used, but may not produce optimal results yet.
            - The Whisper model currently supports only English.
            - The term "equation" denotes the required math mode. This is the only mode implemented so far.

            ### Example Inputs
            Here are some example inputs you can try:
            - *equation x squared plus two x plus one is equal to zero*
            - *equation integral of one over x is equal to the natural log of the absolute value of x plus some constant c*
            - *equation the Cauchy-Riemann condition equations*
            
            The outputs will be displayed in the respective sections on the right. For any issues or bugs, please flag or email me (a screenshot would be appreciated).

            Thank you for trying WhiLa! Have fun!

            **By Brian Welman** - [brianallisterwelman@gmail.com](mailto:brianallisterwelman@gmail.com)
            """
        )
        with gradio.Row():
            interface.render()

    from gradio.routes import mount_gradio_app

    # Launch the Gradio app
    return mount_gradio_app(
        app=web_app,
        blocks=demo,
        path="/",
    )


# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
