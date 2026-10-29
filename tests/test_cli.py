import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from local_mlx_stack.cli import main
from local_mlx_stack.ollama import OllamaClient

class CliTests(unittest.TestCase):
    def test_route_no_model_calls(self):
        with io.StringIO() as output, contextlib.redirect_stdout(output):
            self.assertEqual(main(["route", "Fix Python tests"]), 0)
            self.assertIn('"role": "coder"', output.getvalue())

    def test_opencode_dry_run_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("local_mlx_stack.cli.subprocess.run") as run, io.StringIO() as output, contextlib.redirect_stdout(output):
                self.assertEqual(main(["opencode", "Implement a test", "--project", d]), 0)
                run.assert_not_called()
                self.assertIn("Dry run only", output.getvalue())
