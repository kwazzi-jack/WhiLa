from modal import App

from whila.textify import Textifier


class ModalApp:
    def __init__(self) -> None:
        super()

    def _create_modal_app(self):
        self.app = App()

    def _function_wrap(self, function):
        return self.app.function(function)

    def _create_gradio_interface(self): ...
    def _create_textifier(self):
        self.txty = Textifier()

    def launch(): ...
