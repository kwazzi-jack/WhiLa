import logging.config
import platform
from pathlib import Path

import click
import ollama
import yaml
from ollama import ResponseError

from whila.textify import Textifier

# Load logging configuration
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger("whila")

if __name__ == "__main__":

    # Create textifier
    txty = Textifier()

    # Setup gradio
    import gradio as gr

    gr.Interface(
        fn=txty.textify,
        inputs=gr.Audio(source="microphone", type="numpy"),
        outputs=gr.Textbox(),
    ).launch()

# def _get_host_os():
#     os_platform = platform.system()

#     if os_platform == "Linux":
#         return "Linux"
#     elif os_platform == "Darwin":
#         return "macOS"
#     elif os_platform == "Windows":
#         return "Windows"
#     else:
#         raise ValueError(f"Unsupported operating system: {os_platform}")


# def _download_llama3():
#     # Download llama3-instruct via ollama
#     logger.info("Pulling Llama3...")

#     try:
#         logger.debug("Performing ollama pull on 'llama3-instruct'")
#         response = ollama.pull("llama3-instruct")
#         logger.debug("Response: %s", response)

#     except ResponseError as error:
#         logger.warning("Failed to download llama3-instruct.")
#         return False

#     return True


# def _load_llama3():
#     logger.info("Attempting to load Llama3...")
#     logger.debug("Finding ollama models based on OS")

#     # Match based on https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored
#     match _get_host_os():
#         case "macOS":
#             model_path = Path(r"~/.ollama/models")
#             detected_os = "macOS"
#         case "Linux":
#             model_path = Path(r"/usr/share/ollama/.ollama/models")
#             detected_os = "Linux"
#         case "Windows":
#             model_path = Path(r"C:\Users\%username%\.ollama\models")
#             detected_os = "Windows"
#         case _:
#             raise ValueError(f"Unsupported operating system: {_get_host_os()}")
#     logger.debug(f"OS detected: {detected_os}")
#     logger.debug("Expected model path: %s", model_path)

#     # Expand on user's home directory
#     model_path = model_path.expanduser()
#     logger.debug("User expansion of model path: %s", model_path)

#     # Check if the models folder exists
#     if not model_path.exists():
#         logger.info("No model folder found - pulling model...")

#         # Try to download
#         downloaded = _download_llama3()

#         # Model downloaded successfully
#         if downloaded:
#             logger.info("Model downloaded successfully.")
#         else:
#             logger.info(
#                 "Failed to download model. ",
#                 "You can either rerun the setup command or it will be downloaded during use. "
#                 "Rerunning the setup command is recommended.",
#             )
#             raise click.Abort()

#     # # With models folder, check for model
#     # model_path = model_path / "llama3-instruct"
#     # if not model_path.exists():


# @click.command()
# def setup():
#     """Setup the environment for whila."""
#     logger.info("Starting setup...")
#     try:
#         logger.info("Loading models")
#         _load_llama3()
#         logger.info("Setup complete!")
#     except Exception as e:
#         logger.error(f"An error occurred during setup: {e}")
#         raise click.Abort()
