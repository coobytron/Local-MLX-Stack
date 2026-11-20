"""Benchmark loop: fixed tasks drawn from real work, graded by plain code, one comparison table.

No model judges another model. Every check is a regex, a parse, a compile, a test run,
or a lookup you could repeat by hand, so a score change means the *output* changed.
Each model under test fills all three roles (fast, coder, reasoner), so a result answers
"is this one model enough for everything?".
"""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from itertools import permutations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .core import Stack
from .ollama import OllamaClient, OllamaError, Settings
from .routing import select

FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
CJK = re.compile(r"[一-鿿]")
COMMENT = re.compile(r"//[^\n]*")
STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


@dataclass
class Check:
    name: str
    ok: bool | None  # None = could not be checked here (counted as skipped, not failed)
    note: str = ""


@dataclass(frozen=True)
class Task:
    name: str
    category: str
    prompt: str
    grade: Callable[[str], list[Check]]
    runner: Callable[[OllamaClient], str] | None = None  # default: one chat call with `prompt`
    heavy: bool = False  # several model calls; only runs with --heavy or when named
    tier: str = "easy"  # "hard" tasks only run with --hard or when named


# --- shared helpers ------------------------------------------------------------------

def pick_code(text: str, *, hint: str = "") -> str | None:
    blocks = FENCE.findall(text)
    for lang, code in blocks:
        if hint and (hint in lang.lower() or hint in code.lower()):
            return code
    if blocks:
        return blocks[0][1]
    if hint and hint in text.lower():
        return text  # unfenced reply that still contains the file
    return None


def looped(text: str) -> bool:
    """True when most long lines are repeats, like the 'Work State' summary loop."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
    if len(lines) < 6:
        return False
    repeats = sum(n - 1 for n in Counter(lines).values() if n > 1)
    return repeats / len(lines) > 0.4


def node_parses(js: str) -> bool | None:
    if not shutil.which("node"):
        return None
    run = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(0,'utf8'))"],
                         input=js, capture_output=True, text=True, timeout=15)
    return run.returncode == 0


def js_cases(code: str, cases: list[tuple[str, object]]) -> list[bool | None]:
    """Run the model's JavaScript in Node and compare each call's JSON to the expected value."""
    if not shutil.which("node"):
        return [None] * len(cases)
    code = re.sub(r"^export\s+(default\s+)?", "", code, flags=re.M)
    body = ["const __out=[];"]
    for call, expected in cases:
        want = json.dumps(json.dumps(expected, separators=(",", ":"), ensure_ascii=False), ensure_ascii=False)
        body.append(f"try{{__out.push(JSON.stringify({call})==={want})}}catch(e){{__out.push(false)}}")
    body.append("console.log(JSON.stringify(__out));")
    try:
        run = subprocess.run(["node"], input=code + "\n" + "\n".join(body), capture_output=True,
                             text=True, timeout=15)
        results = json.loads(run.stdout.strip().splitlines()[-1])
        return [bool(r) for r in results] if len(results) == len(cases) else [False] * len(cases)
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return [False] * len(cases)


def js_grader(fn_name: str, cases: list[tuple[str, object]]) -> Callable[[str], list[Check]]:
    def grade(text: str) -> list[Check]:
        code = pick_code(text, hint=fn_name.lower())
        out = [Check(f"defines {fn_name}", code is not None and fn_name in code)]
        if not out[0].ok:
            return out
        out += [Check(call, ok) for (call, _), ok in zip(cases, js_cases(code, cases))]
        return out
    return grade


# --- web: single-file HTML timer ------------------------------------------------------

def grade_timer(text: str) -> list[Check]:
    html = pick_code(text, hint="<html")
    out = [Check("html file found", html is not None)]
    if html is None:
        return out
    low = html.lower()
    out += [
        Check("start, pause and reset present", all(w in low for w in ("start", "pause", "reset"))),
        Check("dark mode follows the system", "prefers-color-scheme" in low),
        Check("no external libraries", not re.search(r'<(script|link)[^>]+(src|href)\s*=\s*["\']https?:', low)),
    ]
    js = "\n".join(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S | re.I))
    parsed = node_parses(js) if js.strip() else False
    out.append(Check("javascript parses", parsed, "" if js.strip() else "no inline script"))
    return out


# --- javascript: pure functions with test cases ---------------------------------------

TITLE_CASES = [
    ('titleCase("the lord of the rings")', "The Lord of the Rings"),
    ('titleCase("late-night jazz to chill")', "Late-Night Jazz to Chill"),
    ('titleCase("a day in the life")', "A Day in the Life"),
    ('titleCase("CHILL vibes")', "Chill Vibes"),
    ('titleCase("going to")', "Going To"),
    ('titleCase("")', ""),
]
WRAP_CASES = [
    ('wrapLines("one two three four", 7)', ["one two", "three", "four"]),
    ('wrapLines("a b c", 100)', ["a b c"]),
    ('wrapLines("", 5)', []),
    ('wrapLines("extraordinarily long word", 5)', ["extraordinarily", "long", "word"]),
    ('wrapLines("hello   world", 11)', ["hello world"]),
]


# --- extraction and summarizing -------------------------------------------------------

NOTES = """Notes from the week:
- Dana Ortiz confirmed for 2026-03-14 (venue TBD)
- Priya Rao asked to move to 03/22/2026
- Left a voicemail for Mateo Silva; he's free March 30, 2026"""
WANT_PEOPLE = {"Dana Ortiz": "2026-03-14", "Priya Rao": "2026-03-22", "Mateo Silva": "2026-03-30"}


def grade_extract(text: str) -> list[Check]:
    blob = pick_code(text) or text
    start, end = blob.find("["), blob.rfind("]")
    try:
        rows = json.loads(blob[start:end + 1])
        got = {r["name"]: r["date"] for r in rows}
    except (ValueError, KeyError, TypeError):
        return [Check("valid JSON list of {name, date}", False)]
    return [
        Check("valid JSON list of {name, date}", True),
        Check("exactly the three people", set(got) == set(WANT_PEOPLE), ", ".join(sorted(got))[:80]),
        Check("dates normalized to YYYY-MM-DD", got == WANT_PEOPLE),
    ]


PARAGRAPH = (
    "Our local model stack runs on a Mac with 48GB of unified memory. It keeps three models available: a 12 "
    "billion parameter model for quick extraction, a 30 billion parameter coder, and a 35 billion parameter "
    "reasoner. Only one model is loaded at a time, and each is unloaded after replying so memory stays free. "
    "A keyword router chooses the model, and a separate pipeline can chain all three for planning work. Testing "
    "showed the small model matched the large one on short web pages, while the large coder was steadier on "
    "multi-step edits.")


def grade_summary(text: str) -> list[Check]:
    bullets = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith(("- ", "* "))]
    numbers = set(re.findall(r"\d+", text))
    return [
        Check("exactly 3 bullets", len(bullets) == 3, f"{len(bullets)} found"),
        Check("each bullet is 14 words or fewer", bool(bullets) and all(len(b.split()) <= 14 for b in bullets)),
        Check("keeps the 48GB fact", "48gb" in text.lower().replace(" ", "")),
        Check("invents no numbers", numbers <= set(re.findall(r"\d+", PARAGRAPH)) | {"1", "2", "3"}),
    ]


# --- swift, metal and embedded: compile or lookup ----------------------------------------

SWIFT_SNIPPET = '''public enum Mode: String {
    case left, right
    public var label: String {
        switch self {
        case .left:  return "左旋转"   // 逆时针
        case .right: return "右旋转"   // 顺时针
        }
    }
}
let lightModes = [(0, "关闭灯光"), (1, "白光"), (2, "流光")]
func status(_ ok: Bool) -> String { ok ? "已恢复" : "回读不一致" }
'''


def grade_translate(text: str) -> list[Check]:
    code = pick_code(text)
    out = [Check("code block found", code is not None)]
    if code is None:
        return out
    leftovers = [s for s in STRING.findall(COMMENT.sub("", code)) if CJK.search(s)]
    out.append(Check("no Chinese left in string literals", not leftovers, " ".join(leftovers[:3])))
    comments = lambda s: [c.strip() for c in COMMENT.findall(s)]
    out.append(Check("comments untouched", comments(code) == comments(SWIFT_SNIPPET)))
    shape = lambda s: re.sub(r"\s+", " ", STRING.sub('""', s)).strip()
    out.append(Check("logic unchanged (only strings differ)", shape(code) == shape(SWIFT_SNIPPET)))
    return out


SWIFT_MAIN = '''print(runLengths("aaabcc").map { "\\($0.0)\\($0.1)" }.joined())
print(runLengths("").count)
print(runLengths("zz").map { "\\($0.0)\\($0.1)" }.joined())
'''


def swift_outputs(code: str, main: str = SWIFT_MAIN) -> tuple[list[str] | None, str]:
    """Compile the model's Swift with a small main.swift and run it. None = no compiler here."""
    if not shutil.which("swiftc"):
        return None, "swiftc not installed"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "lib.swift").write_text(code)
        (Path(d) / "main.swift").write_text(main)
        try:
            build = subprocess.run(["swiftc", "lib.swift", "main.swift", "-o", "t"], cwd=d,
                                   capture_output=True, text=True, timeout=240)
            if build.returncode != 0:
                lines = [ln for ln in build.stderr.splitlines() if "error:" in ln]
                return [], (lines[0] if lines else "compile failed")[-120:]
            run = subprocess.run(["./t"], cwd=d, capture_output=True, text=True, timeout=10)
            return run.stdout.split(), ""
        except subprocess.TimeoutExpired:
            return [], "timed out"


def grade_swift(text: str) -> list[Check]:
    code = pick_code(text, hint="runlengths")
    out = [Check("function found", code is not None and "runLengths" in code)]
    if not out[0].ok:
        return out
    lines, note = swift_outputs(code)
    if lines is None:
        return out + [Check("compiles", None, note), Check("run-length output correct", None, note)]
    return out + [Check("compiles", bool(lines) or not note, note),
                  Check("run-length output correct", lines == ["a3b1c2", "0", "z2"])]


def metal_compiles(code: str) -> tuple[bool | None, str]:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "k.metal"
        src.write_text(code)
        try:
            run = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(src), "-o", str(Path(d) / "k.air")],
                                 capture_output=True, text=True, timeout=90)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None, "metal compiler unavailable"
        if run.returncode != 0 and ("unable to find utility" in run.stderr or "missing Metal Toolchain" in run.stderr):
            return None, "Metal toolchain missing (xcodebuild -downloadComponent MetalToolchain)"
        errors = [ln for ln in run.stderr.splitlines() if "error:" in ln]
        return run.returncode == 0, "" if run.returncode == 0 else (errors[0] if errors else "compile failed")[-120:]


def grade_metal(text: str) -> list[Check]:
    code = pick_code(text, hint="metal_stdlib")
    out = [Check("shader file found", code is not None)]
    if code is None:
        return out
    ok, note = metal_compiles(code)
    return out + [Check("defines fragment vignette", bool(re.search(r"\bfragment\b[^;{]*\bvignette\b", code))),
                  Check("compiles with the Metal compiler", ok, note)]


@functools.lru_cache(maxsize=1)
def m5_names() -> frozenset[str] | None:
    base = Path.home() / "Documents" / "Arduino" / "libraries"
    libs = [d for d in base.glob("M5*") if d.name.startswith(("M5Unified", "M5GFX"))]
    if not libs:
        return None
    names: set[str] = set()
    for lib in libs:
        for f in lib.rglob("*"):
            if f.suffix in (".h", ".hpp"):
                names |= set(re.findall(r"\w+", f.read_text(errors="ignore")))
    return frozenset(names)


def grade_sketch(text: str) -> list[Check]:
    code = pick_code(text, hint="m5unified")
    out = [Check("sketch found", code is not None)]
    if code is None:
        return out
    out += [
        Check("includes M5Unified.h", "M5Unified.h" in code),
        Check("no old M5StickCPlus library", "M5StickCPlus" not in code),
        Check("calls M5.begin and M5.update", "M5.begin(" in code and "M5.update(" in code),
        Check("80 lines or fewer", len(code.strip().splitlines()) <= 80),
    ]
    names = m5_names()
    if names is None:
        out.append(Check("library calls exist", None, "M5Unified not installed"))
    else:
        used = set(re.findall(r"\bM5\.(\w+)\.(\w+)\(", code))
        used_flat = {n for pair in used for n in pair} | set(re.findall(r"\bM5\.(\w+)\(", code))
        missing = sorted(used_flat - names)
        out.append(Check("library calls exist", not missing, ", ".join(missing[:5])))
    return out


# --- reasoning ---------------------------------------------------------------------------

def grade_order(text: str) -> list[Check]:
    last = re.findall(r"ORDER:\s*([A-D](?:\s*,\s*[A-D]){3})", text)
    return [Check("ends with an ORDER line", bool(last)),
            Check("correct order B,C,D,A", bool(last) and last[-1].replace(" ", "") == "B,C,D,A")]


# --- pipeline: stays grounded in the real project ----------------------------------------

FIXTURE = {
    "README.md": "# Tiny CLI\n\nGreets people. Run `python app.py --name Sam`.\n",
    "app.py": ("import argparse\n\n\ndef greet(name):\n    return f\"Hello, {name}!\"\n\n\ndef main():\n"
               "    p = argparse.ArgumentParser()\n    p.add_argument(\"--name\", default=\"world\")\n"
               "    args = p.parse_args()\n    print(greet(args.name))\n\n\nif __name__ == \"__main__\":\n    main()\n"),
    "docs/notes.md": "Keep output to one line. No third-party packages.\n",
}
FIXTURE_NAMES = {"app.py", "readme.md", "notes.md"}
PIPELINE_TASK = "Add a --shout flag to the existing greeter that uppercases the greeting."


def run_pipeline(client: OllamaClient) -> str:
    with tempfile.TemporaryDirectory() as d:
        for rel, body in FIXTURE.items():
            path = Path(d) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        out = Stack(client).pipeline(PIPELINE_TASK, "review", project=Path(d), use_acb=False)
    return "\n\n".join(f"## {k.upper()}\n{out[k]}" for k in ("brief", "plan", "draft", "review"))


def grade_pipeline(text: str) -> list[Check]:
    named = {Path(p).name.lower() for p in re.findall(r"[\w./-]+\.(?:py|js|ts|tsx|json|md|html|swift|toml|ya?ml|sh|txt)\b", text)}
    invented = sorted(named - FIXTURE_NAMES)
    return [
        Check("cites the real app.py", "app.py" in text),
        Check("mentions the --shout flag", "--shout" in text),
        Check("invents no files", not invented, ", ".join(invented[:5])),
        Check("adds no third-party packages", not re.search(r"pip install|import (click|typer|requests)", text)),
    ]


# --- hard tier: tasks a 12B-30B model can plausibly fail --------------------------------

LONG_NOTES = """Offsite logistics thread, compiled 2026-02-27 (budget approval due 04/01/2026):
- Dana Ortiz confirmed, arriving 2026-03-14 (venue TBD)
- Priya Rao asked to move her arrival to 03/22/2026
- Mateo Silva confirmed by voicemail; he lands March 30, 2026
- Hannah Weiss is coming, landing the 4th of April 2026
- Omar Haddad declined (audit conflict); ignore his earlier 2026-03-18
- Lena Fischer confirmed, arrival 2026-04-09, vegetarian
- Reminder: flight refunds close on 05/15/2026 (not an arrival)
- Tomas Reyes will arrive on April 12, 2026 (was 04/10/2026 before the change)
- Kwame Boateng: arriving 03/17/2026, bringing a guest
- Our host, Ines Carvalho, is local and is not traveling
- Yuki Tanaka confirmed for the 21st of March 2026
- Sofia Marin cancelled on 03/03/2026
- Raj Patel confirmed, arriving 2026-03-25
- Final headcount review is March 10, 2026 (not an arrival)
- Catering needs numbers by the 1st of March 2026
- Badge pickup opens 03/13/2026 at the front desk
- Shuttle schedule goes out March 6, 2026
- Anyone not listed above is not coming"""
WANT_LONG = {"Dana Ortiz": "2026-03-14", "Priya Rao": "2026-03-22", "Mateo Silva": "2026-03-30",
             "Hannah Weiss": "2026-04-04", "Lena Fischer": "2026-04-09", "Tomas Reyes": "2026-04-12",
             "Kwame Boateng": "2026-03-17", "Yuki Tanaka": "2026-03-21", "Raj Patel": "2026-03-25"}


def json_rows(text: str) -> list | None:
    blob = pick_code(text) or text
    try:
        return json.loads(blob[blob.find("["):blob.rfind("]") + 1])
    except ValueError:
        return None


def grade_extract_long(text: str) -> list[Check]:
    rows = json_rows(text)
    try:
        got = {r["name"]: r["date"] for r in rows or []}
    except (KeyError, TypeError):
        rows = None
    if rows is None:
        return [Check("valid JSON list of {name, date}", False)]
    return [
        Check("valid JSON list of {name, date}", True),
        Check("only people who are arriving", set(got) == set(WANT_LONG),
              ", ".join(sorted(set(got) ^ set(WANT_LONG)))[:80]),
        Check("every arrival date correct (ISO)", got == WANT_LONG,
              ", ".join(n for n in WANT_LONG if got.get(n) != WANT_LONG[n])[:80]),
    ]


def csv_call(line: str) -> str:
    return f"parseCsvLine({json.dumps(line)})"


CSV_CASES = [
    (csv_call("a,b,c"), ["a", "b", "c"]),
    (csv_call('a,"b,c",d'), ["a", "b,c", "d"]),
    (csv_call('"say ""hi""",x'), ['say "hi"', "x"]),
    (csv_call("a,,c,"), ["a", "", "c", ""]),
    (csv_call(""), [""]),
    (csv_call('"",x'), ["", "x"]),
    (csv_call(" a , b "), [" a ", " b "]),
    (csv_call('"a ""b"", c",d'), ['a "b", c', "d"]),
]
SEMVER_CASES = [
    ('compareSemver("1.2.3", "1.10.0")', -1),
    ('compareSemver("2.0.0", "1.99.99")', 1),
    ('compareSemver("1.0.0", "1.0.0-rc.1")', 1),
    ('compareSemver("1.0.0-alpha", "1.0.0-alpha.1")', -1),
    ('compareSemver("1.0.0-beta.2", "1.0.0-beta.11")', -1),
    ('compareSemver("1.0.0-alpha.1", "1.0.0-alpha.beta")', -1),
    ('compareSemver("1.0.0-rc.1", "1.0.0-beta.11")', 1),
    ('compareSemver("1.0.0+build5", "1.0.0+build9")', 0),
]

RECALL_QUESTION = "What is the current passcode for the cold-storage room in Building C?"
RECALL_ANSWER = "7391-LK"
RECALL_DECOYS = ["4410-QX", "1187-MR", "6205-TV", "7319-LK"]
RECALL_NEEDLE = ("After the second inspection day the passcode for the cold-storage room in Building C "
                 f"was changed to {RECALL_ANSWER}, and the facilities office confirmed it in writing.")
RECALL_NEAR_MISSES = [
    f"The passcode for the cold-storage room in Building B is {RECALL_DECOYS[0]}.",
    f"Before the change, the passcode for the cold-storage room in Building C was {RECALL_DECOYS[1]}.",
    f"The passcode for the dry-storage room in Building C is {RECALL_DECOYS[2]}.",
    f"A draft memo proposed {RECALL_DECOYS[3]} for the cold-storage room in Building C, but it was rejected.",
]


def recall_document() -> str:
    """About 6,000 characters of dull warehouse log with the needle in the middle and four near-misses."""
    crews, docks = ("Alder", "Birch", "Cedar", "Dunn"), ("north", "east", "south", "west")
    goods = ("tinned fruit", "paper rolls", "glass jars", "garden tools", "frozen peas")
    lines = [f"Day {d}: crew {crews[d % 4]} moved {40 + d * 7 % 90} pallets of {goods[d % 5]} to the {docks[d % 4]} "
             f"dock, and the shift log was filed before {8 + d % 9}:{(d * 13) % 60:02d}." for d in range(1, 51)]
    for at, extra in zip((6, 14, 22, 33), RECALL_NEAR_MISSES):
        lines.insert(at, extra)
    lines.insert(len(lines) // 2, RECALL_NEEDLE)
    return "\n".join(lines)


def grade_recall(text: str) -> list[Check]:
    bare = text.strip().strip("`'\".* ")
    return [
        Check("finds the buried passcode", RECALL_ANSWER in text),
        Check("names no near-miss passcode", not any(d in text for d in RECALL_DECOYS)),
        Check("answers with only the passcode", bare == RECALL_ANSWER),
    ]


JSON_KEYS = {"title", "priority", "tags", "estimate_hours", "owner"}


def grade_json_strict(text: str) -> list[Check]:
    bare = "reply is one bare JSON object (no fences, no prose)"
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return [Check(bare, False)]
    if not isinstance(obj, dict):
        return [Check(bare, False)]
    title, tags, hours = obj.get("title"), obj.get("tags"), obj.get("estimate_hours")
    return [
        Check(bare, True),
        Check("exactly the five keys", set(obj) == JSON_KEYS, ", ".join(sorted(obj))[:80]),
        Check("title is 10-40 characters and mentions login",
              isinstance(title, str) and 10 <= len(title) <= 40 and "login" in title.lower()),
        Check('priority is "high"', obj.get("priority") == "high"),
        Check("tags are 3 distinct lowercase words of 3-12 letters",
              isinstance(tags, list) and len(tags) == 3 and len(set(map(str, tags))) == 3
              and all(isinstance(t, str) and re.fullmatch(r"[a-z]{3,12}", t) for t in tags)),
        Check("estimate_hours is an integer from 1 to 40", type(hours) is int and 1 <= hours <= 40),
        Check("owner is null", "owner" in obj and obj["owner"] is None),
    ]


RUNNERS = ("Ana", "Ben", "Cal", "Dee", "Eli")
LOGIC_CLUES = [
    ("Ben did not finish directly next to Ana.", lambda o: abs(o.index("Ben") - o.index("Ana")) != 1),
    ("Cal finished somewhere behind Dee.", lambda o: o.index("Cal") > o.index("Dee")),
    ("Eli finished either first or last.", lambda o: o.index("Eli") in (0, 4)),
    ("Dee finished exactly two places ahead of Ben.", lambda o: o.index("Dee") + 2 == o.index("Ben")),
    ("Ana finished somewhere ahead of Eli.", lambda o: o.index("Ana") < o.index("Eli")),
    ("Cal finished somewhere ahead of Ben.", lambda o: o.index("Cal") < o.index("Ben")),
]
LOGIC_PROMPT = ("Five runners, Ana, Ben, Cal, Dee and Eli, finished a race with no ties. Clues:\n"
                + "\n".join(f"{i}. {t}" for i, (t, _) in enumerate(LOGIC_CLUES, 1))
                + "\nWork out the finishing order, first place to last. End your reply with one line exactly like "
                  "'ORDER: Name,Name,Name,Name,Name'.")


def logic_solutions() -> list[tuple[str, ...]]:
    return [o for o in permutations(RUNNERS) if all(ok(o) for _, ok in LOGIC_CLUES)]


def grade_logic(text: str) -> list[Check]:
    lines = re.findall(r"ORDER:\s*([A-Za-z]+(?:\s*,\s*[A-Za-z]+){4})", text)
    order = tuple(n.strip() for n in lines[-1].split(",")) if lines else ()
    valid = sorted(order) == sorted(RUNNERS)
    return [
        Check("ends with an ORDER line of the five runners", valid),
        Check("satisfies every clue", valid and all(ok(order) for _, ok in LOGIC_CLUES)),
        Check("matches the unique solution", order in logic_solutions()),
    ]


GROUP_MAIN = '''func line<T>(_ pairs: [(T, Int)]) -> String { pairs.isEmpty ? "0" : pairs.map { "\\($0.0)x\\($0.1)" }.joined(separator: ",") }
print(line(groupedCounts(["b", "a", "b", "c", "a", "b"])))
print(line(groupedCounts([3, 1, 3, 2, 1, 2, 2])))
print(line(groupedCounts(Array("hello"))))
print(line(groupedCounts([Int]())))
print(line(groupedCounts([true, false, false])))
'''
GROUP_EXPECTED = ["bx3,ax2,cx1", "2x3,3x2,1x2", "lx2,hx1,ex1,ox1", "0", "falsex2,truex1"]


def grade_swift_generic(text: str) -> list[Check]:
    code = pick_code(text, hint="groupedcounts")
    out = [Check("function found", code is not None and "groupedCounts" in code)]
    if not out[0].ok:
        return out
    out.append(Check("declared generic over Hashable",
                     bool(re.search(r"func\s+groupedCounts\s*<\s*\w+\s*:\s*Hashable", code))))
    lines, note = swift_outputs(code, GROUP_MAIN)
    if lines is None:
        return out + [Check("compiles", None, note), Check("counts and tie order correct", None, note)]
    return out + [Check("compiles", bool(lines) or not note, note),
                  Check("counts and tie order correct", lines == GROUP_EXPECTED, " ".join(lines)[:80])]


MEMO = (
    "The Port Authority of Aldermere approved the Harbor Gate renovation on February 9 after eleven months of "
    "review. The project replaces the aging ferry terminal on the east quay, which was built in 1974 and now "
    "handles about 1,800 passengers on a typical weekday. Engineers found that the main concourse roof had lost "
    "much of its load margin, and that three of the six loading ramps could no longer be certified for heavy "
    "vehicles. Closing the terminal outright was rejected because commuters from the island towns have no other "
    "crossing within forty kilometers.\n\n"
    "Instead, construction will happen in three phases. Phase one, starting March 12, rebuilds the two western "
    "ramps and adds a temporary waiting hall beside the car park. Ferry service continues from the remaining "
    "ramps, though the sailing interval stretches from twenty minutes to thirty. Phase two replaces the concourse "
    "roof and is scheduled for the quieter winter months, when passenger numbers fall by roughly a third. Phase "
    "three rebuilds the ticket hall and adds step-free access to every platform, a requirement the old building "
    "never met.\n\n"
    "The authority has set the budget at 2.4 million dollars, of which the regional transport fund covers 1.5 "
    "million and the remainder comes from port fees. Councillor Mara Ivanova, who chaired the review panel, "
    "argued that the fee increase is modest compared with the cost of an emergency closure. Opponents, led by the "
    "island ferry users' association, say the thirty minute interval will push some commuters back to cars and "
    "want a guarantee that fares will not rise during construction. The authority has promised to freeze fares "
    "until phase three is complete but declined to put the promise in writing, saying that fuel costs are "
    "outside its control.\n\n"
    "Contractor selection closes on February 28. Bidders must show experience with working marine structures, "
    "and the authority will weigh safety record at least as heavily as price. Local firms have been encouraged "
    "to join consortia, and the authority says it expects about forty construction jobs to be filled from the "
    "region. A public meeting at the Harbor Gate hall on February 20 will let residents see the plans and "
    "question the engineers. Notes from the meeting will be published within a week, together with the final "
    "phasing calendar.\n\n"
    "Environmental monitoring is also part of the approval. Divers will survey the seabed before any piling work, "
    "and a noise limit applies during the spawning season for the local herring. If the limit is exceeded on "
    "three separate days, work must pause until the contractor submits a revised method. The authority will "
    "publish monitoring results monthly.")
MEMO_FACTS = {"March 12": r"march\s+12\b", "2.4 million": r"2\.4\s+million", "February 28": r"february\s+28\b"}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def grade_summary_strict(text: str) -> list[Check]:
    bullets = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith(("- ", "* "))]
    source = ngrams(MEMO, 6)
    copied = [b for b in bullets if ngrams(b, 6) & source]
    missing = [name for name, rx in MEMO_FACTS.items() if not re.search(rx, text, re.I)]
    return [
        Check("exactly 4 bullets", len(bullets) == 4, f"{len(bullets)} found"),
        Check("each bullet is 18 words or fewer", bool(bullets) and all(len(b.split()) <= 18 for b in bullets)),
        Check("keeps the three required facts", not missing, ", ".join(missing)),
        Check("invents no numbers", set(NUMBER.findall(text)) <= set(NUMBER.findall(MEMO)) | {"1", "2", "3", "4"}),
        Check("copies no run of 6 words from the text", not copied, copied[0][:60] if copied else ""),
        Check("nothing but the bullets", len([ln for ln in text.splitlines() if ln.strip()]) == len(bullets)),
    ]


TASKS = [
    Task("timer_html", "web",
         "Write one self-contained index.html: a focus timer with a big countdown, start/pause/reset "
         "buttons, and a dark mode that follows the system. No libraries. Reply with the file in one html code block.",
         grade_timer),
    Task("js_titlecase", "javascript",
         "Write a JavaScript function titleCase(str). Capitalize every word and lowercase the rest, but keep these "
         "small words lowercase unless first or last: a, an, and, the, of, to, in, on, at, for, with. Hyphenated "
         "words capitalize each part. Input may be any case. Reply with only the function in one js code block.",
         js_grader("titleCase", TITLE_CASES)),
    Task("js_wrap", "javascript",
         "Write a JavaScript function wrapLines(text, max) that wraps text greedily into an array of lines of at "
         "most max characters, never splitting a word (a word longer than max gets its own line), collapsing runs of "
         "spaces, and returning [] for empty text. Reply with only the function in one js code block.",
         js_grader("wrapLines", WRAP_CASES)),
    Task("extract_json", "extraction",
         "Extract every person and their date from these notes as a JSON array of objects with keys name and "
         "date (ISO YYYY-MM-DD). Reply with only the JSON.\n\n" + NOTES,
         grade_extract),
    Task("summarize_3", "summarizing",
         "Summarize this in exactly 3 bullet points, each starting with '- ' and at most 14 words. Do not add facts "
         "or numbers that are not in the text. Reply with only the bullets.\n\n" + PARAGRAPH,
         grade_summary),
    Task("translate", "swift",
         "Translate the Chinese string literals in this Swift code into English. Do not translate comments, "
         "do not change anything else. Glossary: 左旋转 Rotate Left, 右旋转 Rotate Right. "
         "Reply with the full code in one swift code block.\n\n```swift\n" + SWIFT_SNIPPET + "```",
         grade_translate),
    Task("swift_func", "swift",
         "Write a Swift function `func runLengths(_ s: String) -> [(Character, Int)]` that run-length encodes a "
         "string, so \"aaabcc\" gives a3, b1, c2 as (Character, Int) pairs. Only the function, no top-level code. "
         "Reply with it in one swift code block.",
         grade_swift),
    Task("metal_shader", "metal",
         "Write a complete Metal shading language file: #include <metal_stdlib>, a struct VertexOut with "
         "float4 position [[position]] and float2 uv, and a fragment function named vignette taking "
         "(VertexOut in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]) that samples "
         "the texture and darkens toward the edges by distance from uv (0.5, 0.5). Reply with the file in one metal code block.",
         grade_metal),
    Task("m5_sketch", "embedded",
         "Write one Arduino sketch for the M5StickC Plus (135x240) using M5Unified. Show battery percent and a "
         "live uptime counter. Button A cycles screen brightness through 3 levels. Button B toggles portrait and "
         "landscape. Under 80 lines. Reply with the sketch in one cpp code block.",
         grade_sketch),
    Task("reason_order", "reasoning",
         "Four tasks A, B, C, D must be done one at a time. B must come before A. C cannot be first. D must come "
         "immediately after C. A must be last. Work it out, then end your reply with one line exactly like "
         "'ORDER: X,Y,Z,W'.",
         grade_order),
    Task("pipeline_grounded", "pipeline", PIPELINE_TASK, grade_pipeline, runner=run_pipeline, heavy=True),
    # hard tier: opt-in with --hard
    Task("hard_extract", "extraction",
         "From this thread, extract every person who is travelling to the offsite and their arrival date, as a JSON "
         "array of objects with keys name and date (ISO YYYY-MM-DD). Skip people who declined, cancelled or are not "
         "travelling, and skip dates that are not an arrival date. Reply with only the JSON.\n\n" + LONG_NOTES,
         grade_extract_long, tier="hard"),
    Task("hard_csv", "javascript",
         "Write a JavaScript function parseCsvLine(line) that splits one CSV line into an array of strings. Fields "
         "may be wrapped in double quotes; inside quotes a comma is literal and two double quotes mean one. Keep "
         "spaces as they are, keep empty fields (including a trailing one), and return [\"\"] for an empty line. "
         "Reply with only the function in one js code block.",
         js_grader("parseCsvLine", CSV_CASES), tier="hard"),
    Task("hard_semver", "javascript",
         "Write a JavaScript function compareSemver(a, b) that returns -1, 0 or 1 following semver precedence for "
         "strings like 1.2.3, 1.0.0-rc.1 or 1.0.0+build5. A version with a pre-release is lower than the same version "
         "without one. Pre-release identifiers split on dots: numeric ones compare as numbers, others compare as "
         "text, a numeric identifier is lower than a text one, and a shorter list is lower when all earlier "
         "identifiers are equal. Ignore build metadata after +. Reply with only the function in one js code block.",
         js_grader("compareSemver", SEMVER_CASES), tier="hard"),
    Task("hard_recall", "long-context",
         "Read this log and answer the question that follows it. Reply with only the passcode.\n\n"
         + recall_document() + "\n\nQuestion: " + RECALL_QUESTION,
         grade_recall, tier="hard"),
    Task("hard_json", "format",
         "Describe a bug ticket for a login timeout that blocks every customer. Reply with one JSON object and "
         "nothing else (no code fence, no commentary) with exactly these keys: title (string, 10 to 40 characters, "
         "mentions login), priority (one of \"low\", \"medium\", \"high\"; pick the one that fits a bug blocking "
         "every customer), tags (array of exactly 3 distinct lowercase words, 3 to 12 letters each), "
         "estimate_hours (integer from 1 to 40, not a string), owner (null).",
         grade_json_strict, tier="hard"),
    Task("hard_logic", "reasoning", LOGIC_PROMPT, grade_logic, tier="hard"),
    Task("hard_swift", "swift",
         "Write a Swift generic function `func groupedCounts<T: Hashable>(_ items: [T]) -> [(T, Int)]` that counts "
         "equal items and returns the (item, count) pairs sorted by count, highest first. Items with the same count "
         "keep the order in which they first appeared in the input. An empty array gives an empty result. Only the "
         "function, no top-level code. Reply with it in one swift code block.",
         grade_swift_generic, tier="hard"),
    Task("hard_summary", "summarizing",
         "Summarize this memo in exactly 4 bullet points, each starting with '- ' and at most 18 words. Keep the "
         "start date of phase one, the budget, and the contractor deadline. Do not add facts or numbers that are "
         "not in the text, and do not copy any run of 6 or more consecutive words from it. Reply with only the "
         "bullets.\n\n" + MEMO,
         grade_summary_strict, tier="hard"),
]

TASKS_BY_NAME = {t.name: t for t in TASKS}

ROUTER_CASES = [
    ("Write a Metal shader", "coder"), ("Fix the bug in my SwiftUI view", "coder"),
    ("Add a unit test for the parser", "coder"), ("Refactor this function", "coder"),
    ("Compare tradeoffs of two caching strategies", "reasoner"), ("Plan a migration to Conductor", "reasoner"),
    ("Design the architecture for an asset index", "reasoner"), ("Audit the risks of this rollout", "reasoner"),
    ("Summarize this paragraph", "fast"), ("Extract the dates from this email", "fast"),
    ("What is the capital of France", "fast"), ("Rewrite this sentence to be friendlier", "fast"),
]


def router_accuracy() -> tuple[int, int, list[str]]:
    """Free check, no model: does the keyword router send typical prompts to the right role?"""
    misses = [f"{p!r} -> {select(p).role} (wanted {want})" for p, want in ROUTER_CASES if select(p).role != want]
    return len(ROUTER_CASES) - len(misses), len(ROUTER_CASES), misses


# --- runner ---------------------------------------------------------------------------------

def run(settings: Settings, models: list[str], tasks: list[Task], *,
        client_factory=OllamaClient, progress: Callable[[str], None] = lambda s: None) -> list[dict]:
    results = []
    for model in models:
        client = client_factory(replace(settings, fast=model, coder=model, reasoner=model))
        for task in tasks:
            progress(f"{model} · {task.name} ...")
            start, error, text = time.monotonic(), None, ""
            try:
                text = task.runner(client) if task.runner else client.chat("coder", task.prompt, "")
            except OllamaError as exc:
                error = str(exc)
            seconds = round(time.monotonic() - start, 1)
            checks = [Check("model answered", False, error)] if error else (
                task.grade(text) + [Check("no repetition loop", not looped(text))])
            results.append({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model, "task": task.name, "category": task.category, "seconds": seconds,
                "chars": len(text), "passed": sum(c.ok is True for c in checks),
                "failed": sum(c.ok is False for c in checks), "skipped": sum(c.ok is None for c in checks),
                "checks": [vars(c) for c in checks], "response": text[:20000],
            })
            progress(f"  {results[-1]['passed']}/{results[-1]['passed'] + results[-1]['failed']} in {seconds}s")
    return results


def format_table(results: list[dict]) -> str:
    models = list(dict.fromkeys(r["model"] for r in results))
    tasks = list(dict.fromkeys(r["task"] for r in results))
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        cells.setdefault((r["model"], r["task"]), []).append(r)
    repeated = any(len(g) > 1 for g in cells.values())

    def signature(r: dict) -> tuple:
        """Same score AND same failing checks, so two different 3-pass runs still count as flaky."""
        return r["passed"], frozenset(c["name"] for c in r["checks"] if c["ok"] is False)

    def summed(rows: list[dict]) -> str:
        good = sum(r["passed"] for r in rows)
        text = f"{good}/{good + sum(r['failed'] for r in rows)}"
        return text + (f" x{len(rows)}" if repeated else "")

    def cell_text(rows: list[dict]) -> str:
        flaky = len({signature(r) for r in rows}) > 1
        return f"{summed(rows)} {sum(r['seconds'] for r in rows) / len(rows):.0f}s" + ("*" if flaky else "")

    texts = {k: cell_text(g) for k, g in cells.items()}
    tw, mw = max(len(t) for t in tasks + ["total"]), max(len(m) for m in models + ["00/00 000s"])
    if repeated:
        mw = max([mw] + [len(t) for t in texts.values()])
    lines = ["  ".join([f"{'task':<{tw}}"] + [f"{m:>{mw}}" for m in models])]
    for t in tasks:
        lines.append("  ".join([f"{t:<{tw}}"] + [texts.get((m, t), "-").rjust(mw) for m in models]))
    totals = []
    for m in models:
        mine = [r for r in results if r["model"] == m]
        runs = max(len(g) for (cm, _), g in cells.items() if cm == m)
        good = sum(r["passed"] for r in mine)
        count = f" x{runs}" if repeated else ""
        totals.append(f"{good}/{good + sum(r['failed'] for r in mine)}{count} "
                      f"{sum(r['seconds'] for r in mine) / runs:.0f}s".rjust(mw))
    lines += ["-" * len(lines[0]), "  ".join([f"{'total':<{tw}}"] + totals)]
    if repeated:
        lines.append("pass/graded summed over runs, xN = runs, seconds = mean per run")
        if any(len({signature(r) for r in g}) > 1 for g in cells.values()):
            lines.append("* results differed between runs")
    misses: dict[tuple[str, str, str], list] = {}  # (model, task, check) -> [failing runs, note, notes differ]
    for r in results:
        for c in r["checks"]:
            if c["ok"] is False:
                entry = misses.setdefault((r["model"], r["task"], c["name"]), [0, c["note"], False])
                entry[0] += 1
                if c["note"] and entry[1] and c["note"] != entry[1]:
                    entry[2] = True
                elif c["note"] and not entry[1]:
                    entry[1] = c["note"]
    miss_lines = [f"  {m} · {t}: {n}" + (f" ({note}{'; notes vary' if varied else ''})" if note else "")
                  + (f" ({fails}/{len(cells[(m, t)])} runs)" if len(cells[(m, t)]) > 1 else "")
                  for (m, t, n), (fails, note, varied) in misses.items()]
    skipped = sorted({c["name"] for r in results for c in r["checks"] if c["ok"] is None})
    if miss_lines:
        lines += ["", "Failed checks:"] + miss_lines
    if skipped:
        lines += ["", "Not checkable here: " + ", ".join(skipped)]
    return "\n".join(lines)


def save(results: list[dict], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
