"""Agentic benchmark: does OpenCode, driven by a local model, actually finish real edit jobs?

The chat bench cannot see stalls (re-reading the same files, repeating the auto-summary template,
never editing). Each task here runs `opencode run` inside a FRESH temporary copy of a small fixture
project, then plain code grades the result by running it. No model judges another model.
Nothing is ever written outside the temp directory.
"""
from __future__ import annotations

import hashlib
import os
import py_compile
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .bench import Check, looped
from .ollama import Settings
from .opencode import CONFIG, plan

WARNING = "Agentic bench runs OpenCode with edit permissions inside temporary copies only."
SUMMARY_HEADER = "## Objective"
CJK = re.compile(r"[一-鿿]")
COMMENT = re.compile(r"#[^\n]*")
STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


class AgenticError(ValueError):
    pass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    seconds: float
    timed_out: bool = False


Runner = Callable[[str, str, Path, int], RunResult]  # (model, prompt, workdir, timeout) -> RunResult


@dataclass(frozen=True)
class AgenticTask:
    name: str
    prompt: str
    files: dict[str, str]
    grade: Callable[[Path, RunResult], list[Check]]
    timeout: int = 300
    allowed_new: tuple[str, ...] = ()  # new files the task may legitimately create (relative paths)
    category: str = "agentic"


# --- running things safely -------------------------------------------------------------

def run_group(cmd: list[str], cwd: Path, timeout: float, env: dict | None = None) -> RunResult:
    """Run argv (no shell) in its own process group; on timeout kill the whole group."""
    # OpenCode trusts $PWD over the real cwd; an inherited PWD made it work in the launching
    # shell's folder instead of the temp copy. Always point PWD at where we actually run.
    env = {**(os.environ if env is None else env), "PWD": str(cwd)}
    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = proc.communicate()
    return RunResult(proc.returncode if proc.returncode is not None else -1, out or "",
                     round(time.monotonic() - start, 1), timed_out)


def make_opencode_runner(settings: Settings) -> Runner:
    """The real runner. Refuses cleanly before doing anything if opencode is missing."""
    if not shutil.which("opencode"):
        raise AgenticError("`opencode` is not on PATH; the agentic bench needs OpenCode installed.")

    def runner(model: str, prompt: str, workdir: Path, timeout: int) -> RunResult:
        p = plan(prompt, workdir, "coder", replace(settings, fast=model, coder=model, reasoner=model))
        env = os.environ.copy()
        if p["config"]:
            env["OPENCODE_CONFIG"] = p["config"]
        elif CONFIG.exists():
            env["OPENCODE_CONFIG"] = str(CONFIG)
        return run_group(p["command"], workdir, timeout, env)
    return runner


def py(workdir: Path, args: list[str], timeout: int = 15) -> RunResult:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return run_group([sys.executable, *args], workdir, timeout, env)


# --- fixture helpers -------------------------------------------------------------------

def write_fixture(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def ignored(rel: Path) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in rel.parts)


def snapshot(root: Path) -> dict[str, str]:
    """relative path -> sha256, skipping opencode's own dot-folders and bytecode."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not ignored(rel):
            out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def read(workdir: Path, rel: str) -> str:
    try:
        return (workdir / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def universal_checks(task: AgenticTask, workdir: Path, run: RunResult, before: dict[str, str]) -> list[Check]:
    after = snapshot(workdir)
    changed = sorted(p for p in after if before.get(p) != after[p]) + sorted(p for p in before if p not in after)
    stray = sorted(p for p in after if p not in before and p not in task.allowed_new)
    stdout = run.stdout or ""
    headers = stdout.count(SUMMARY_HEADER)
    return [
        Check("finished before timeout", not run.timed_out, f"limit {task.timeout}s" if run.timed_out else ""),
        Check("no summary loop", headers <= 2 and not looped(stdout),
              f"{headers} summary header(s)" if headers > 2 else ""),
        # qwen3-coder sometimes prints a tool call as plain text ('<function=grep>') and stops.
        Check("tool calls executed, not printed as text", "<function=" not in stdout),
        Check("changed at least one file", bool(changed), ", ".join(changed[:3])),
        Check("touched only files in the fixture", not stray, ", ".join(stray[:3])),
    ]


# --- task 1: add a flag ----------------------------------------------------------------

APP = '''import argparse


def greet(name):
    return f"Hello, {name}!"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="world")
    args = p.parse_args()
    print(greet(args.name))


if __name__ == "__main__":
    main()
'''


def grade_add_flag(workdir: Path, run: RunResult) -> list[Check]:
    plain = py(workdir, ["app.py", "--name", "sam"])
    shout = py(workdir, ["app.py", "--name", "sam", "--shout"])
    return [
        Check("plain output unchanged", plain.returncode == 0 and plain.stdout.strip() == "Hello, sam!",
              plain.stdout.strip()[:60]),
        Check("--shout uppercases the greeting", shout.returncode == 0 and shout.stdout.strip() == "HELLO, SAM!",
              shout.stdout.strip()[-60:]),
    ]


# --- task 2: fix an off-by-one ---------------------------------------------------------

CALC = '''def total_up_to(n):
    """Sum of the integers 1 through n, inclusive."""
    return sum(range(n))
'''
TEST_CALC = '''import unittest

from calc import total_up_to


class TotalTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(total_up_to(0), 0)

    def test_three(self):
        self.assertEqual(total_up_to(3), 6)


if __name__ == "__main__":
    unittest.main()
'''


def grade_fix_bug(workdir: Path, run: RunResult) -> list[Check]:
    tests = py(workdir, ["-m", "unittest"])
    return [
        Check("unit tests pass", tests.returncode == 0 and not tests.timed_out, tests.stdout.strip()[-80:]),
        Check("test file untouched", read(workdir, "test_calc.py") == TEST_CALC),
    ]


# --- task 3: translate string literals -------------------------------------------------

LABELS = '''"""Status labels for the light controller."""
MODES = {
    "left": "左旋转",   # 逆时针
    "right": "右旋转",  # 顺时针
}

LIGHTS = [(0, "关闭灯光"), (1, "白光"), (2, "流光")]


def label(mode):
    # 找不到时返回默认值
    return MODES.get(mode, "未知模式")


def status(ok):
    if ok:
        return "已恢复"
    return "回读不一致"


def describe(index):
    for i, text in LIGHTS:
        if i == index:
            return "灯光：" + text
    return "没有这个灯光"


def report(mode, ok):
    # 汇总输出
    return label(mode) + " / " + status(ok)
'''


def grade_translate_file(workdir: Path, run: RunResult) -> list[Check]:
    code = read(workdir, "labels.py")
    comments = lambda s: [c.strip() for c in COMMENT.findall(s)]
    shape = lambda s: re.sub(r"\s+", " ", STRING.sub('""', s)).strip()
    # Drop the module docstring from the string scan by working on non-comment code only.
    leftovers = [s for s in STRING.findall(COMMENT.sub("", code)) if CJK.search(s)]
    try:
        with tempfile.TemporaryDirectory() as scratch:
            py_compile.compile(str(workdir / "labels.py"), cfile=str(Path(scratch) / "labels.pyc"), doraise=True)
        compiles, note = True, ""
    except (py_compile.PyCompileError, OSError, ValueError) as exc:
        compiles, note = False, str(exc)[:80]
    return [
        Check("file still present", bool(code)),
        Check("no Chinese left in string literals", bool(code) and not leftovers, " ".join(leftovers[:3])),
        Check("comments untouched", comments(code) == comments(LABELS)),
        Check("logic unchanged (only strings differ)", shape(code) == shape(LABELS)),
        Check("file still compiles", compiles, note),
    ]


# --- task 4: rename across files (the multi-file stall test) ---------------------------

RENAME_FILES = {
    "main.py": ('from pricing.invoice import invoice_line\nfrom reports.summary import summarize\n'
                'from reports.export import export_csv\nfrom reports.cart import cart_total\n\n'
                'print(invoice_line([2, 3, 5]))\nprint(summarize([1, 2, 3]))\n'
                'print(export_csv([4, 4]))\nprint(cart_total([10, 20]))\n'),
    "pricing/core.py": 'def calc_total(items):\n    """Sum of all item prices."""\n    return sum(items)\n',
    "pricing/invoice.py": ('from pricing.core import calc_total\n\n\ndef invoice_line(items):\n'
                           '    return f"invoice: {calc_total(items)}"\n'),
    "reports/summary.py": ('from pricing.core import calc_total\n\n\ndef summarize(items):\n'
                           '    return f"summary: {calc_total(items)} over {len(items)}"\n'),
    "reports/export.py": ('from pricing.core import calc_total\n\n\ndef export_csv(items):\n'
                          '    return ",".join(str(i) for i in items) + "," + str(calc_total(items))\n'),
    "reports/cart.py": ('from pricing.core import calc_total\n\n\ndef cart_total(items):\n'
                        '    return calc_total(items) * 2\n'),
}
RENAME_USERS = ["pricing/core.py", "pricing/invoice.py", "reports/summary.py", "reports/export.py",
                "reports/cart.py"]
RENAME_OUTPUT = "invoice: 10\nsummary: 6 over 3\n4,4,8\n60"


def grade_rename_multi(workdir: Path, run: RunResult) -> list[Check]:
    leftover = [rel for rel in RENAME_FILES if "calc_total" in read(workdir, rel)]
    missing = [rel for rel in RENAME_USERS if "compute_total" not in read(workdir, rel)]
    main = py(workdir, ["main.py"])
    return [
        Check("no calc_total left anywhere", not leftover, ", ".join(leftover[:3])),
        Check("compute_total in all 5 files", not missing, "missing: " + ", ".join(missing[:3]) if missing else ""),
        Check("main.py prints the same output", main.returncode == 0 and main.stdout.strip() == RENAME_OUTPUT,
              main.stdout.strip()[-80:].replace("\n", " | ")),
    ]


TASKS = [
    AgenticTask("add_flag", "Add a --shout flag to app.py that uppercases the greeting. Keep the default output unchanged.",
                {"app.py": APP}, grade_add_flag),
    AgenticTask("fix_bug", "The unit tests in this project fail. Fix the bug in the code so `python3 -m unittest` "
                "passes. Do not edit the test file.",
                {"calc.py": CALC, "test_calc.py": TEST_CALC}, grade_fix_bug),
    AgenticTask("translate_file", "In labels.py translate only the Chinese string literals into English. Leave all "
                "comments (still Chinese) and all code logic exactly as they are.",
                {"labels.py": LABELS}, grade_translate_file),
    AgenticTask("rename_multi", "Rename the function calc_total to compute_total everywhere in this project: its "
                "definition and every place it is used, across all folders. Do not change behavior.",
                RENAME_FILES, grade_rename_multi, timeout=420),
]


# --- runner ----------------------------------------------------------------------------

def run_agentic(settings: Settings, models: list[str], tasks: list[AgenticTask], *,
                runner: Runner | None = None, timeout: int | None = None,
                progress: Callable[[str], None] = lambda s: None) -> list[dict]:
    if runner is None:
        runner = make_opencode_runner(settings)  # raises AgenticError before anything is created
    results = []
    for model in models:
        for task in tasks:
            limit = timeout or task.timeout
            progress(f"{model} · {task.name} ...")
            with tempfile.TemporaryDirectory(prefix="stack-agentic-") as d:
                workdir = Path(d).resolve()
                write_fixture(workdir, task.files)
                before = snapshot(workdir)
                try:
                    run = runner(model, task.prompt, workdir, limit)
                except (OSError, AgenticError) as exc:
                    run = RunResult(-1, f"runner failed: {exc}", 0.0)
                checks = universal_checks(replace(task, timeout=limit), workdir, run, before) + task.grade(workdir, run)
            results.append({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model, "task": task.name, "category": task.category, "seconds": run.seconds,
                "chars": len(run.stdout), "passed": sum(c.ok is True for c in checks),
                "failed": sum(c.ok is False for c in checks), "skipped": sum(c.ok is None for c in checks),
                "checks": [vars(c) for c in checks], "stdout": run.stdout[:20000],
            })
            progress(f"  {results[-1]['passed']}/{results[-1]['passed'] + results[-1]['failed']} in {run.seconds}s")
    return results
