from whila.appify import CONVERT_TEMPLATE, EXTRACT_TEMPLATE, Latexifier


class TestExtractor:
    latexifier = Latexifier()

    def test_extractor(self):
        text = "equation mode with e to the power of minus "
        "eye pi then plus one equals naught I think"
        expected = EXTRACT_TEMPLATE.format(input_text=text)
        output = self.latexifier.extractor(text)
        assert output == expected

    def test_cleanup_on_extract(self):
        text = 'COMMAND="equation", '
        'VALUE="e to the power of minus eye pi then plus one equals naught I think'
        output = self.latexifier.cleanup("extract", text)
        expected = (
            "equation",
            "e to the power of minus eye pi then plus one equals naught I think",
        )
        assert output == expected

    def test_cleanup_on_error(self):
        text = 'ERROR="invalid-command", '
        'VALUE="e to the power of minus eye pi then plus one equals naught I think'
        output = self.latexifier.cleanup("extract", text)
        expected = (
            "equation",
            "e to the power of minus eye pi then plus one equals naught I think",
        )
        assert output == expected
