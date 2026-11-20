import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from local_mlx_stack import agentic, bench
from local_mlx_stack.agentic import RunResult
from local_mlx_stack.cli import main


def row(model, task, passed, failed=0, seconds=6.0, run=None, misses=()):
    checks = [{"name": n, "ok": False, "note": note} for n, note in misses]
    out = {"model": model, "task": task, "passed": passed, "failed": failed, "seconds": seconds, "checks": checks}
    if run:
        out["run"] = run
    return out


class RepeatTableTests(unittest.TestCase):
    def test_single_run_output_is_unchanged_with_or_without_run_key(self):
        plain = [row("m", "t", 3, 1, 4.0, misses=[("c", "n")])]
        tagged = [{**plain[0], "run": 1}]
        text = bench.format_table(plain)
        self.assertEqual(text, bench.format_table(tagged))
        self.assertNotIn("pass/graded", text)
        self.assertNotIn("*", text)
        self.assertIn("  m · t: c (n)\n"[:-1], text)
        self.assertNotIn("runs)", text)
        self.assertEqual(text.splitlines()[1], "t          3/4 4s")

    def test_three_runs_are_summed_with_mean_seconds(self):
        rows = [row("m", "t", 6, 0, s, run=i) for i, s in ((1, 4.0), (2, 6.0), (3, 8.0))]
        text = bench.format_table(rows)
        self.assertIn("18/18 x3 6s", text)
        self.assertEqual(text.splitlines()[-1], "pass/graded summed over runs, xN = runs, seconds = mean per run")
        self.assertNotIn("differed", text)
        self.assertNotIn("*", text)
        widths = {len(ln) for ln in text.splitlines()[:4]}
        self.assertEqual(len(widths), 1)  # header, row, rule and total stay aligned

    def test_flaky_marker_only_when_runs_differ(self):
        rows = [row("m", "t", 8, 0, run=1), row("m", "t", 7, 1, run=2), row("m", "t", 8, 0, run=3),
                row("m", "steady", 4, 0, run=1), row("m", "steady", 4, 0, run=2), row("m", "steady", 4, 0, run=3)]
        text = bench.format_table(rows)
        lines = text.splitlines()
        self.assertTrue(next(ln for ln in lines if ln.startswith("t ")).endswith("23/24 x3 6s*"))
        self.assertFalse(next(ln for ln in lines if ln.startswith("steady")).endswith("*"))
        self.assertIn("* results differed between runs", lines)
        self.assertIn("35/36 x3 12s", lines[4])  # total row sums the same way

    def test_failed_checks_collapse_with_counts(self):
        miss = ("no calc_total left anywhere", "")
        rows = [row("q", "rename_multi", 7, 1, run=1, misses=[miss]),
                row("q", "rename_multi", 8, 0, run=2),
                row("q", "rename_multi", 7, 1, run=3, misses=[miss, ("other", "first")]),
                ]
        rows[2]["checks"][1]["note"] = "first"
        text = bench.format_table(rows)
        self.assertEqual(text.count("no calc_total left anywhere"), 1)
        self.assertIn("  q · rename_multi: no calc_total left anywhere (2/3 runs)", text)
        self.assertIn("  q · rename_multi: other (first) (1/3 runs)", text)


class RepeatCliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["bench", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_out_of_range_repeat_is_rejected(self):
        for bad in ("0", "11"):
            code, _, err = self.run_cli("--repeat", bad)
            self.assertEqual(code, 2)
            self.assertIn("Error: --repeat must be between 1 and 10", err)

    def test_chat_bench_repeats_and_saves_every_run(self):
        calls = []

        def fake_run(settings, models, tasks, progress=None, **kw):
            calls.append(1)
            return [row("m", "t", 2, 0)]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(bench, "run", side_effect=fake_run):
            path = Path(d) / "out.jsonl"
            code, out, err = self.run_cli("--tasks", "translate", "--repeat", "3", "--output", str(path))
            lines = [json.loads(ln) for ln in path.read_text().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual([r["run"] for r in lines], [1, 2, 3])
        self.assertIn("6/6 x3", out)
        self.assertIn("run 2/3", err)

    def test_default_repeat_adds_no_notation(self):
        with mock.patch.object(bench, "run", return_value=[row("m", "t", 2, 0)]):
            _, out, err = self.run_cli("--tasks", "translate")
        self.assertNotIn("pass/graded", out)
        self.assertNotIn("run 1", err)

    def test_agentic_path_honors_repeat(self):
        seen = []

        def fake(model, prompt, workdir, timeout):
            seen.append(model)
            return RunResult(0, "", 0.1)
        with mock.patch.object(agentic, "make_opencode_runner", return_value=fake):
            code, out, err = self.run_cli("--agentic", "--models", "a", "--tasks", "add_flag", "--repeat", "2")
        self.assertEqual(code, 0)
        self.assertEqual(len(seen), 2)
        self.assertIn("x2", out)
        self.assertIn("run 2/2", err)
