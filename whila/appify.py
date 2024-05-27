from pathlib import Path

import gradio
import modal

VOLUME = {"/whila": modal.Volume.from_name("whila-volume")}
MODEL_DIR = Path("/whila/models")
TTS_MODEL = "openai/whisper-medium.en"
TTL_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
PIP_PACKAGES = ["vllm", "torch", "transformers", "gradio"]
PY_VERSION = "3.10"
IMAGE = modal.Image.debian_slim(python_version=PY_VERSION).pip_install(*PIP_PACKAGES)
GPU_CONFIG = modal.gpu.A10G()

EXTRACT_TEMPLATE = r"""
**Extraction Task:**

1. **Commands:**
   - The input must begin with a command from the following list:
     ```
     commands = [equation, align, next, end]
     ```
   - The command must be at the start of the input text and must match exactly.

2. **Values:**
   - The value follows the command and represents a mathematical expression, theorem, or result.
   - Capture the value as it is, with minimal editing, and focus on the mathematical object being described.
   - The only editing allowed is to remove conversational aspects not related to the description.
   - If a value cannot be identified, return "invalid-value".

3. **Input and Output:**
   - Input will be provided in the format:
     ```
     INPUT="<input_text>"
     ```
   - Output should be structured as:
     ```
     COMMAND="<command_identified>", VALUE="<value_identified>"
     ```
   - If an invalid command or value is found, return the error in the format:
     ```
     ERROR="<invalid_error>"
     ```

4. **Example Input:**
    ```
    INPUT="equation uhm ex square with plus why is equal ah to ... zero"
    ```

5. **Example Output:**
    ```
    COMMAND="equation", VALUE="ex square plus why equal to zero"
    ```

Given the above, here is the input:

INPUT="{input_text}"
"""

CONVERT_TEMPLATE = r"""
**Conversion Task:**

1. **Input:**
   - The input will be provided in the format:
     ```
     VALUE="<value_text>"
     ```

2. **Output:**
   - Convert the given `VALUE` text into valid LaTeX code that can be used in math-mode.
   - You are not required to insert inline or multiline math-mode symbols, only the contents for such an environment.
   - The LaTeX code should be formatted nicely and use normal conventions for writing mathematics.
   - The output should be an exact conversion of what is being described in the `VALUE` text without any corrections or creative liberties.
   - If the conversion cannot be done, return "invalid-conversion".

3. **Example Input:**
    ```
    VALUE="sum of two to the power of eye calculated from eye equals one to big n"
    ```

4. **Example Output:**
    ```
    \sum\limits^N_{i = 1} 2^i
    ```

Given the above, here is the input:

VALUE="{value_text}"
"""

app = modal.App(
    "whila-app",
    image=IMAGE,
    volumes=VOLUME,
)


# refactor this into a separate file, modalize
@app.cls(gpu=GPU_CONFIG)
class Latexifier:  # use vllm probably

    @modal.enter()
    def load_model(self):
        import vllm

        self.extract_template = EXTRACT_TEMPLATE
        self.convert_template = CONVERT_TEMPLATE
        self.model_params = vllm.SamplingParams(temperature=0.1, top_p=0.95)
        # Create TTL LLM agent
        self.ttl = vllm.LLM(TTL_MODEL, tensor_parallel_size=GPU_CONFIG.count)

        # Save point for `align`
        self.save = ""

        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L108-L111

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

    def _get_text_from_value(self, text):
        """Extract the value out of the formatted output from LLM"""
        return text.split("=")[1][1:-1]

    def cleanup(self, exec_type, output):
        """Clean-up the output from the LLM based on the execution
        type."""

        # If an error was produced, raise it
        if "ERROR" in output:
            error = self._get_text_from_value(output).capitalize()
            raise ValueError(f"Error from extractor: '{error}'")

        # Check if it contains a command or a value field
        if "COMMAND" not in output and "VALUE" not in output:
            raise ValueError(f"Cannot find command or value in '{output}'")

        # Output is from extractor, return command and value
        if exec_type == "extract":
            command, value = output.split(",")
            command, value = command.strip(), value.strip()

            if command >= value:
                raise ValueError(f"Incorrect ordering of extractor ouput in '{output}'")

            command = self._get_text_from_value(command)
            value = self._get_text_from_value(value)
            return command, value

        # Output is from converter, return latex
        elif exec_type == "convert":
            return self._get_text_from_value(value)

        # Unknown executor type
        else:
            raise ValueError(f"Unknown executor type `{exec_type}`")

    @modal.method()
    def extract(self, text):
        """Perform extraction of the command and value from the LLM based on
        input audio."""
        extractor = self.extractor(text)
        output = self.ttl.generate(extractor, sampling_params=self.model_params)
        return self.cleanup("extract", output)

    @modal.method()
    def convert(self, value):
        """Perform conversion of the value from the extractor using the LLM
        to get the corresponding latex."""
        converter = self.converter(value)
        output = self.ttl.generate(converter, sampling_params=self.model_params)
        return self.cleanup("convert", output)

    def finalize(self, command, latex):
        """Based on the command, identify and format how the latex should be wrapped"""
        match command.lower():

            # Equation environment
            case "equation":
                return r"\begin{equation}\n\t{contents}\n\end{equation}".format(
                    contents=latex
                )

            # Align (start) environment
            case "align":
                self.save = r"\begin{align}\n\t{contents}".format(
                    contents=latex + r"{contents}"
                )
                return self.save

            # Align (next) environment
            case "next":

                if "contents" not in self.save:
                    raise ValueError(
                        "No align text identified. Please start with 'align'"
                    )

                self.save = self.save.format(contents=latex + r"\\\n\t{contents}")
                return self.save

            # Align (end) environment
            case "end":

                if "contents" not in self.save:
                    raise ValueError(
                        "No align text identified. Please start with 'align'"
                    )
                output = self.save.format(contents=latex + r".\n\end{align}")
                self.save = ""
                return output

            # Error: Unknown environment
            case _:
                raise ValueError(f"Unknown command type '{command}'")

    @modal.method()
    def inference(self, text):
        # see: https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/06_gpu_and_ml/llm-serving/vllm_inference.py#L118-L131

        command, value = self.extract(text)
        latex = self.convert(value)
        output = self.finalize(command, latex)

        return output


# @app.cls(gpu="a10g")
# class Textifier:

#     def __init__(self, tts_model: str):
#         """
#         Initializes the Textifier object with a specified TTS model.

#         Parameters:
#             tts_model (str): The TTS model to be used for text-to-speech.
#         """

#         # Selects first GPU by default, otherwise use CPU
#         device_id = 0 if torch.cuda.is_available() else None

#         # Create pipeline
#         self.transcriber = transformers.pipeline(
#             "automatic-speech-recognition",  # See [1]
#             model=tts_model,
#             framework="pt",  # Using PyTorch
#             device=device_id,
#         )

#     def _transcribe(self, sample_rate: int, raw_data: np.array) -> str:
#         """
#         Transcribe the raw audio data using the provided sample rate and return the transcribed text.

#         Parameters:
#             sample_rate (int): The sample rate of the audio data.
#             raw_data (np.array): The raw audio data to transcribe.

#         Returns:
#             str: The transcribed text from the audio data.
#         """
#         try:
#             result = self.transcriber({"sampling_rate": sample_rate, "raw": raw_data})
#             return result["text"]
#         except Exception as e:
#             print(f"Error during transcription: {e}")
#             return ""

#     def _normalise(self, raw_data):
#         """
#         A function to normalise audio data by maximum value.

#         Parameters:
#             raw_data: The raw audio data to be normalised.

#         Returns:
#             The normalised raw audio data.
#         """
#         raw_data = raw_data.astype(np.float32)
#         if np.max(np.abs(raw_data)) != 0:
#             raw_data /= np.max(np.abs(raw_data))
#         return raw_data

#     def textify(self, raw_audio):
#         """
#         Transcribe the raw audio data using the provided sample rate and return the transcribed text.

#         Parameters:
#             raw_audio: A tuple containing the sample rate and raw audio data.

#         Returns:
#             The transcribed text from the audio data.
#         """
#         sample_rate, raw_data = raw_audio
#         raw_data = self._normalise(raw_data)
#         return self._transcribe(sample_rate, raw_data)

#     def to_gradio(self):
#         """
#         A function to convert the Textifier object to a Gradio interface compatible function.
#         """
#         return lambda raw_audio: self.textify(raw_audio)


class GradioApp:

    def __init__(self):
        self.latexifier = modal.Cls.lookup("whila-app", "Latexifer")
        self._create_gradio_interface()

    def _create_gradio_interface(self):
        self.interface = gradio.Interface(
            self._audio_to_latex(),
            inputs=["text"],
            outputs=["text"],
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
    # GradioApp().launch(share=True)
    latexifier = modal.Cls.lookup("whila-app", "Latexifier")

    output = latexifier.inference.remote(
        "equation cosine brackets of pie divided by two plus two theta close brackets"
    )

    print(output)


# Footnotes:
# [1] https://modal.com/docs/examples/vllm_inference
