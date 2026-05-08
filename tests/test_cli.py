import io
import unittest
from contextlib import redirect_stdout

from marketfm.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_exposes_marketfm_program(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.prog, "marketfm")

    def test_help_smoke(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--help"])

        self.assertEqual(caught.exception.code, 0)

    def test_version_command_prints_version(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--version"])

        self.assertEqual(exit_code, 0)
        self.assertIn("0.1.0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
