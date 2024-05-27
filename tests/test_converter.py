import pytest

from whila.appify import CONVERT_TEMPLATE, Latexifier


class TestConverter:
    latexifier = Latexifier()

    def test_converter(self):
        text = "equation mode with e to the power of minus "
        "eye pi then plus one equals naught I think"
        expected = CONVERT_TEMPLATE.format(input_text=text)
        output = self.latexifier.converter(text)
        assert output == expected

    def test_cleanup_on_extract(self):
        text = (
            "COMMAND=equation, "
            + "VALUE=e to the power of minus eye pi then plus one equals naught I think"
        )
        output = self.latexifier.cleanup("extract", text)
        expected = (
            "equation",
            "e to the power of minus eye pi then plus one equals naught I think",
        )
        assert output == expected

    def test_cleanup_on_command_error(self):
        text = "ERROR=invalid-command"
        expected = "Error from extractor: 'Invalid-command'"
        with pytest.raises(ValueError) as error_info:
            self.latexifier.cleanup("extract", text)

        assert str(error_info.value) == expected

    def test_cleanup_on_value_error(self):
        text = "ERROR=invalid-value"
        expected = "Error from extractor: 'Invalid-value'"
        with pytest.raises(ValueError) as error_info:
            self.latexifier.cleanup("extract", text)

        assert str(error_info.value) == expected
