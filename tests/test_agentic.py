import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from local_mlx_stack import agentic
from local_mlx_stack.agentic import (APP, CALC, LABELS, RENAME_FILES, RunResult, TASKS, AgenticError,
                                     run_agentic, snapshot, write_fixture)
from local_mlx_stack.cli import main
from local_mlx_stack.ollama import Settings

TASK = {t.name: t for t in TASKS}

LABEL_FIXES = [("左旋转", "Rotate Left"), ("右旋转", "Rotate Right"), ("关闭灯光", "Lights Off"), ("白光", "White"),
               ("流光", "Flowing"), ("未知模式", "Unknown"), ("已恢复", "Restored"), ("回读不一致", "Mismatch"),
               ("没有这个灯光", "No such light")]


def translated():
    out = LABELS
    for zh, en in LABEL_FIXES:
        out = out.replace(f'"{zh}"', f'"{en}"')
    return out.replace('"灯光："', '"Light: "')


def solution(name: str) -> dict[str, str]:
    if name == "add_flag":
        return {"app.py": APP.replace('    args = p.parse_args()\n    print(greet(args.name))',
                                      '    p.add_argument("--shout", action="store_true")\n    args = p.parse_args()\n'
                                      '    out = greet(args.name)\n    print(out.upper() if args.shout else out)')}
    if name == "fix_bug":
        return {"calc.py": CALC.replace("range(n)", "range(n + 1)")}
    if name == "translate_file":
        return {"labels.py": translated()}
    return {rel: body.replace("calc_total", "compute_total") for rel, body in RENAME_FILES.items()}


def runner_writing(files_for, stdout="done", timed_out=False):
    def runner(model, prompt, workdir, timeout):
        for rel, body in files_for(workdir).items():
            (workdir / rel).write_text(body, encoding="utf-8")
        return RunResult(0, stdout, 1.0, timed_out)
    return runner


def solver(workdir):
    name = next(n for n, t in TASK.items() if (workdir / next(iter(t.files))).exists()
                and all((workdir / f).exists() for f in t.files))
    return solution(name)


def one(name, runner):
    return run_agentic(Settings(), ["m"], [TASK[name]], runner=runner)[0]


def checks(row):
    return {c["name"]: c["ok"] for c in row["checks"]}


def correct_runner_for(name):
    return runner_writing(lambda wd: solution(name))


class AgenticTests(unittest.TestCase):
    def test_every_task_passes_with_a_correct_solution(self):
        for name in TASK:
            row = one(name, correct_runner_for(name))
            self.assertEqual(row["failed"], 0, (name, [c for c in row["checks"] if c["ok"] is not True]))
            self.assertGreaterEqual(row["passed"], 6, name)
            self.assertEqual(row["category"], "agentic")

    def test_untouched_fixtures_fail_their_graders(self):
        for name in TASK:
            row = one(name, lambda m, p, wd, t: RunResult(0, "ok", 1.0))
            got = checks(row)
            self.assertFalse(got["changed at least one file"], name)
            self.assertGreater(row["failed"], 1, name)

    def test_summary_loop_is_detected(self):
        block = "## Objective\nAdd the flag\n## Work State\nRead app.py again and again\n## Next Move\nEdit app.py now\n"
        row = one("add_flag", runner_writing(lambda wd: solution("add_flag"), stdout=block * 6))
        self.assertFalse(checks(row)["no summary loop"])
        self.assertTrue(checks(row)["changed at least one file"])

    def test_timeout_is_detected(self):
        row = one("add_flag", runner_writing(lambda wd: {}, timed_out=True))
        self.assertFalse(checks(row)["finished before timeout"])

    def test_half_finished_rename_fails(self):
        half = {rel: body.replace("calc_total", "compute_total") for rel, body in RENAME_FILES.items()
                if rel in ("pricing/core.py", "pricing/invoice.py")}
        row = one("rename_multi", runner_writing(lambda wd: half))
        got = checks(row)
        self.assertFalse(got["no calc_total left anywhere"])
        self.assertFalse(got["compute_total in all 5 files"])
        self.assertTrue(got["changed at least one file"])

    def test_edited_test_file_and_stray_files_fail(self):
        files = {"calc.py": solution("fix_bug")["calc.py"], "test_calc.py": "pass\n", "extra.txt": "x"}
        got = checks(one("fix_bug", runner_writing(lambda wd: files)))
        self.assertFalse(got["test file untouched"])
        self.assertFalse(got["touched only files in the fixture"])

    def test_translation_that_changes_a_comment_fails(self):
        bad = translated().replace("# 逆时针", "# counter-clockwise")
        got = checks(one("translate_file", runner_writing(lambda wd: {"labels.py": bad})))
        self.assertFalse(got["comments untouched"])
        self.assertTrue(got["no Chinese left in string literals"])

    def test_dot_folders_are_ignored(self):
        def runner(model, prompt, workdir, timeout):
            (workdir / ".opencode").mkdir()
            (workdir / ".opencode" / "state.json").write_text("{}")
            (workdir / "app.py").write_text(solution("add_flag")["app.py"])
            return RunResult(0, "ok", 1.0)
        self.assertTrue(checks(one("add_flag", runner))["touched only files in the fixture"])

    def test_runs_in_a_temp_copy_and_leaves_nothing_behind(self):
        seen = []

        def runner(model, prompt, workdir, timeout):
            seen.append((workdir, timeout))
            return RunResult(0, "", 0.0)
        run_agentic(Settings(), ["m"], [TASK["add_flag"]], runner=runner, timeout=7)
        self.assertEqual(seen[0][1], 7)
        self.assertFalse(seen[0][0].exists())
        self.assertNotEqual(seen[0][0], Path.cwd())

    def test_runner_that_raises_is_scored_not_crashed(self):
        def runner(*a):
            raise OSError("boom")
        row = one("add_flag", runner)
        self.assertFalse(checks(row)["changed at least one file"])

    def test_default_runner_refuses_without_opencode(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(AgenticError):
                agentic.make_opencode_runner(Settings())
            with self.assertRaises(AgenticError):
                run_agentic(Settings(), ["m"], [TASK["add_flag"]])

    def test_real_process_group_runner_times_out(self):
        with tempfile.TemporaryDirectory() as d:
            res = agentic.run_group(["python3", "-c", "import time; time.sleep(30)"], Path(d), 0.5)
        self.assertTrue(res.timed_out)

    def test_default_runner_builds_the_opencode_command(self):
        with mock.patch("shutil.which", return_value="/bin/opencode"), \
                mock.patch.object(agentic, "run_group", return_value=RunResult(0, "", 0.0)) as rg:
            with tempfile.TemporaryDirectory() as d:
                agentic.make_opencode_runner(Settings())("some-model", "do it", Path(d), 5)
        cmd = rg.call_args.args[0]
        self.assertEqual(cmd, ["opencode", "run", "--model", "ollama/some-model", "do it"])

    def test_cli_agentic_path(self):
        fake = lambda model, prompt, wd, t: RunResult(0, "", 0.1)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(agentic, "make_opencode_runner", return_value=fake), \
                redirect_stdout(out), redirect_stderr(err):
            code = main(["bench", "--agentic", "--models", "a", "--tasks", "add_flag,fix_bug", "--timeout", "9"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("temporary copies only", text)
        self.assertNotIn("Router", text)
        self.assertIn("add_flag", text)
        self.assertIn("Failed checks", text)

    def test_cli_agentic_list_and_unknown(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["bench", "--agentic", "--list"]), 0)
        self.assertIn("rename_multi", out.getvalue())
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["bench", "--agentic", "--tasks", "nope"]), 2)

    def test_cli_agentic_without_opencode_exits_cleanly(self):
        with mock.patch("shutil.which", return_value=None), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["bench", "--agentic", "--tasks", "add_flag"]), 2)
    def test_printed_tool_call_is_flagged(self):
        # Real failure seen with qwen3-coder: the call is printed as text and nothing runs.
        def leaky(model, prompt, workdir, timeout):
            return RunResult(0, "I'll search.\n<function=grep>\n<parameter=pattern>\nx\n</parameter>\n</function>", 1.0)
        rows = run_agentic(Settings(), ["m"], [TASK["add_flag"]], runner=leaky)
        failed = [c["name"] for c in rows[0]["checks"] if c["ok"] is False]
        self.assertIn("tool calls executed, not printed as text", failed)

    def test_run_group_points_pwd_at_the_workdir_not_the_launching_shell(self):
        # Regression: OpenCode used an inherited $PWD and edited/read the wrong folder.
        import os
        with tempfile.TemporaryDirectory() as d:
            run = agentic.run_group(["/bin/sh", "-c", 'echo "$PWD"'], Path(d), 10,
                                    {**os.environ, "PWD": "/somewhere/else"})
        self.assertEqual(run.stdout.strip(), str(Path(d)))


if __name__ == "__main__":
    unittest.main()
