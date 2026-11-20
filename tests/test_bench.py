import json
import unittest
from itertools import permutations
from unittest import mock

from local_mlx_stack import bench
from local_mlx_stack.bench import SWIFT_SNIPPET, Check, Task
from local_mlx_stack.cli import main
from local_mlx_stack.ollama import OllamaError, Settings

GOOD_HTML = """```html
<!DOCTYPE html><html><head><style>@media (prefers-color-scheme: dark){body{background:#000}}</style></head>
<body><button>Start</button><button>Pause</button><button>Reset</button>
<script>let t=1500; function tick(){ t--; }</script></body></html>
```"""

TRANSLATED = (SWIFT_SNIPPET.replace("左旋转", "Rotate Left").replace("右旋转", "Rotate Right")
              .replace("关闭灯光", "Lights Off").replace("白光", "White").replace("流光", "Flowing Light")
              .replace("已恢复", "Restored").replace("回读不一致", "Mismatch"))

SKETCH = """```cpp
#include <M5Unified.h>
void setup(){ M5.begin(); M5.Display.setBrightness(80); }
void loop(){ M5.update(); if (M5.BtnA.wasPressed()) M5.Display.setRotation(1); }
```"""

TITLECASE_JS = """```js
function titleCase(str) {
  const small = new Set(["a","an","and","the","of","to","in","on","at","for","with"]);
  const words = str.toLowerCase().split(" ").filter(Boolean);
  return words.map((w, i) => {
    const cap = (x) => x.split("-").map(p => p.charAt(0).toUpperCase() + p.slice(1)).join("-");
    return (i === 0 || i === words.length - 1 || !small.has(w)) ? cap(w) : w;
  }).join(" ");
}
```"""

HARD_EXTRACT = json.dumps([{"name": n, "date": d} for n, d in bench.WANT_LONG.items()])

CSV_JS = """```js
function parseCsvLine(line) {
  const out = [];
  let cur = "", quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') quoted = false;
      else cur += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { out.push(cur); cur = ""; }
    else cur += c;
  }
  out.push(cur);
  return out;
}
```"""
CSV_NAIVE_JS = "```js\nfunction parseCsvLine(line) { return line.split(','); }\n```"

SEMVER_JS = """```js
function compareSemver(a, b) {
  const split = (v) => { const [core, pre] = v.split("+")[0].split(/-(.*)/s); return [core.split(".").map(Number), pre ? pre.split(".") : []]; };
  const [ca, pa] = split(a), [cb, pb] = split(b);
  for (let i = 0; i < 3; i++) if (ca[i] !== cb[i]) return ca[i] < cb[i] ? -1 : 1;
  if (!pa.length || !pb.length) return pa.length === pb.length ? 0 : pa.length ? -1 : 1;
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    const x = pa[i], y = pb[i], nx = /^\\d+$/.test(x), ny = /^\\d+$/.test(y);
    if (x === y) continue;
    if (nx && ny) return Number(x) < Number(y) ? -1 : 1;
    if (nx !== ny) return nx ? -1 : 1;
    return x < y ? -1 : 1;
  }
  return pa.length === pb.length ? 0 : pa.length < pb.length ? -1 : 1;
}
```"""
# plausible wrong answer: compares pre-release as plain strings, so beta.2 > beta.11 and 1.0.0 < 1.0.0-rc.1
SEMVER_STRING_JS = """```js
function compareSemver(a, b) {
  const x = a.split("+")[0], y = b.split("+")[0];
  return x < y ? -1 : x > y ? 1 : 0;
}
```"""

RECALL_DOC = bench.recall_document()
JSON_GOOD = json.dumps({"title": "Login timeout blocks users", "priority": "high",
                        "tags": ["auth", "timeout", "outage"], "estimate_hours": 8, "owner": None})

LOGIC_GOOD = "Ana is ahead of Eli, Dee leads Ben by two...\nORDER: Ana,Dee,Cal,Ben,Eli"

GROUP_SWIFT = """```swift
func groupedCounts<T: Hashable>(_ items: [T]) -> [(T, Int)] {
    var counts: [T: Int] = [:]
    var firstSeen: [T] = []
    for item in items {
        if counts[item] == nil { firstSeen.append(item) }
        counts[item, default: 0] += 1
    }
    let rank = Dictionary(uniqueKeysWithValues: firstSeen.enumerated().map { ($1, $0) })
    return firstSeen.map { ($0, counts[$0]!) }.sorted {
        $0.1 != $1.1 ? $0.1 > $1.1 : rank[$0.0]! < rank[$1.0]!
    }
}
```"""
# plausible wrong answer: ties broken by text of the item, not by first appearance
GROUP_TIE_WRONG = GROUP_SWIFT.replace("rank[$0.0]! < rank[$1.0]!", "String(describing: $0.0) < String(describing: $1.0)")

SUMMARY_GOOD = """- Harbor Gate ferry terminal gets a phased rebuild, starting March 12 with two western ramps.
- Budget is 2.4 million dollars, split between a regional fund and port fees.
- Ferry sailings slow from twenty to thirty minutes while crews work; opponents want fares frozen in writing.
- Contractor bids close February 28; safety record counts as much as price."""


def by_name(checks):
    return {c.name: c.ok for c in checks}


class GraderTests(unittest.TestCase):
    def test_loop_detector_flags_repeated_summaries(self):
        summary = "\n".join(f"- Completed the focus timer application item number {i % 2}" for i in range(12))
        self.assertTrue(bench.looped(summary))
        self.assertFalse(bench.looped("\n".join(f"A different long line of prose number {i} here" for i in range(12))))

    def test_timer_grader(self):
        good = by_name(bench.grade_timer(GOOD_HTML))
        self.assertTrue(all(good[k] for k in ("html file found", "start, pause and reset present",
                                              "dark mode follows the system", "no external libraries")))
        self.assertIn(good["javascript parses"], (True, None))  # None only if node is missing
        bad = by_name(bench.grade_timer("```html\n<html><script src='https://cdn.x/lib.js'></script></html>\n```"))
        self.assertFalse(bad["no external libraries"])
        self.assertFalse(bad["dark mode follows the system"])
        self.assertFalse(bench.grade_timer("no code here")[0].ok)

    def test_javascript_functions_are_run_against_cases(self):
        results = by_name(bench.js_grader("titleCase", bench.TITLE_CASES)(TITLECASE_JS))
        self.assertTrue(results["defines titleCase"])
        cases = [v for k, v in results.items() if k.startswith("titleCase(")]
        self.assertTrue(all(v in (True, None) for v in cases), results)
        wrong = by_name(bench.js_grader("titleCase", bench.TITLE_CASES)("```js\nfunction titleCase(s){return s}\n```"))
        if bench.shutil.which("node"):
            self.assertFalse(wrong['titleCase("the lord of the rings")'])
        self.assertFalse(bench.js_grader("titleCase", bench.TITLE_CASES)("```js\nconst x = 1\n```")[0].ok)

    def test_extract_grader(self):
        good = json.dumps([{"name": n, "date": d} for n, d in bench.WANT_PEOPLE.items()])
        self.assertTrue(all(c.ok for c in bench.grade_extract(f"```json\n{good}\n```")))
        wrong = json.dumps([{"name": "Dana Ortiz", "date": "03/14/2026"}])
        checks = by_name(bench.grade_extract(wrong))
        self.assertFalse(checks["exactly the three people"])
        self.assertFalse(bench.grade_extract("not json")[0].ok)

    def test_summary_grader(self):
        good = "- Runs on a 48GB Mac with three models\n- One model loads at a time\n- Router picks the model"
        self.assertTrue(all(c.ok for c in bench.grade_summary(good)))
        bad = by_name(bench.grade_summary("- one\n- two\n- three\n- four 99"))
        self.assertFalse(bad["exactly 3 bullets"])
        self.assertFalse(bad["keeps the 48GB fact"])
        self.assertFalse(bad["invents no numbers"])

    def test_translate_grader(self):
        good = by_name(bench.grade_translate(f"```swift\n{TRANSLATED}```"))
        self.assertTrue(all(good.values()), good)
        left = by_name(bench.grade_translate(f"```swift\n{SWIFT_SNIPPET}```"))
        self.assertFalse(left["no Chinese left in string literals"])
        edited = by_name(bench.grade_translate(f"```swift\n{TRANSLATED.replace('ok ?', 'ok &&')}```"))
        self.assertFalse(edited["logic unchanged (only strings differ)"])
        no_comments = by_name(bench.grade_translate(f"```swift\n{TRANSLATED.replace('// 逆时针', '// counter')}```"))
        self.assertFalse(no_comments["comments untouched"])

    def test_swift_and_metal_graders_use_the_compilers(self):
        swift = "```swift\nfunc runLengths(_ s: String) -> [(Character, Int)] { [] }\n```"
        with mock.patch.object(bench, "swift_outputs", return_value=(["a3b1c2", "0", "z2"], "")):
            self.assertTrue(by_name(bench.grade_swift(swift))["run-length output correct"])
        with mock.patch.object(bench, "swift_outputs", return_value=([], "error: nope")):
            self.assertFalse(by_name(bench.grade_swift(swift))["run-length output correct"])
        with mock.patch.object(bench, "swift_outputs", return_value=(None, "swiftc not installed")):
            self.assertIsNone(by_name(bench.grade_swift(swift))["compiles"])
        shader = "```metal\n#include <metal_stdlib>\nfragment float4 vignette() { return 0; }\n```"
        with mock.patch.object(bench, "metal_compiles", return_value=(True, "")):
            self.assertTrue(by_name(bench.grade_metal(shader))["compiles with the Metal compiler"])
        with mock.patch.object(bench, "metal_compiles", return_value=(False, "error: x")):
            self.assertFalse(by_name(bench.grade_metal(shader))["compiles with the Metal compiler"])

    def test_missing_metal_toolchain_is_skipped_not_failed(self):
        gone = mock.Mock(returncode=1, stderr="error: cannot execute tool 'metal' due to missing Metal Toolchain")
        with mock.patch.object(bench.subprocess, "run", return_value=gone):
            ok, note = bench.metal_compiles("kernel void k() {}")
        self.assertIsNone(ok)
        self.assertIn("MetalToolchain", note)

    def test_sketch_grader_checks_library_names(self):
        names = frozenset({"M5", "begin", "update", "Display", "setBrightness", "setRotation", "BtnA", "wasPressed"})
        with mock.patch.object(bench, "m5_names", return_value=names):
            self.assertTrue(by_name(bench.grade_sketch(SKETCH))["library calls exist"])
            invented = SKETCH.replace("setBrightness", "setScreenGlow")
            checks = bench.grade_sketch(invented)
            self.assertFalse(by_name(checks)["library calls exist"])
            self.assertIn("setScreenGlow", [c for c in checks if c.name == "library calls exist"][0].note)
        with mock.patch.object(bench, "m5_names", return_value=None):
            self.assertIsNone(by_name(bench.grade_sketch(SKETCH))["library calls exist"])

    def test_order_and_pipeline_graders(self):
        self.assertTrue(all(c.ok for c in bench.grade_order("Reasoning...\nORDER: B,C,D,A")))
        self.assertFalse(bench.grade_order("ORDER: A,B,C,D")[1].ok)
        grounded = "Edit app.py to add --shout. README.md already documents the flag usage."
        self.assertTrue(all(c.ok for c in bench.grade_pipeline(grounded)))
        made_up = by_name(bench.grade_pipeline("Change app.py and create server.js for --shout"))
        self.assertFalse(made_up["invents no files"])

    def test_router_accuracy_is_free_and_reports_misses(self):
        right, total, misses = bench.router_accuracy()
        self.assertEqual(total, len(bench.ROUTER_CASES))
        self.assertEqual(right + len(misses), total)


class HardGraderTests(unittest.TestCase):
    def assertAllPass(self, checks):
        self.assertTrue(checks and all(c.ok is True for c in checks), by_name(checks))

    def test_hard_tasks_are_tagged_and_easy_ones_unchanged(self):
        hard = [t for t in bench.TASKS if t.tier == "hard"]
        self.assertEqual(len(hard), 8)
        self.assertTrue(all(t.name.startswith("hard_") and not t.heavy for t in hard))
        self.assertTrue(all(t.tier == "easy" for t in bench.TASKS if not t.name.startswith("hard_")))
        self.assertEqual(Task("t", "c", "p", lambda s: []).tier, "easy")

    def test_long_extraction_needs_every_arrival_and_no_distractors(self):
        self.assertGreaterEqual(len(bench.LONG_NOTES.splitlines()), 18)
        self.assertAllPass(bench.grade_extract_long(f"```json\n{HARD_EXTRACT}\n```"))
        rows = lambda d: json.dumps([{"name": n, "date": v} for n, v in d.items()])
        with_declined = by_name(bench.grade_extract_long(rows({**bench.WANT_LONG, "Omar Haddad": "2026-03-18"})))
        self.assertFalse(with_declined["only people who are arriving"])
        old_date = by_name(bench.grade_extract_long(rows({**bench.WANT_LONG, "Tomas Reyes": "2026-04-10"})))
        self.assertFalse(old_date["every arrival date correct (ISO)"])
        us_format = by_name(bench.grade_extract_long(rows({**bench.WANT_LONG, "Priya Rao": "03/22/2026"})))
        self.assertFalse(us_format["every arrival date correct (ISO)"])
        self.assertFalse(bench.grade_extract_long("none")[0].ok)

    def test_csv_and_semver_graders_run_in_node(self):
        for grader, good, wrong in ((bench.TASKS_BY_NAME["hard_csv"].grade, CSV_JS, CSV_NAIVE_JS),
                                    (bench.TASKS_BY_NAME["hard_semver"].grade, SEMVER_JS, SEMVER_STRING_JS)):
            checks = grader(good)
            self.assertGreaterEqual(len(checks) - 1, 6)
            self.assertTrue(all(c.ok in (True, None) for c in checks), by_name(checks))
            if bench.shutil.which("node"):
                self.assertTrue(all(c.ok is True for c in checks))
                self.assertTrue(any(c.ok is False for c in grader(wrong)))

    def test_recall_document_is_long_and_has_one_real_answer(self):
        self.assertTrue(5500 <= len(RECALL_DOC) <= 7000, len(RECALL_DOC))
        self.assertEqual(RECALL_DOC.count(bench.RECALL_ANSWER), 1)
        self.assertTrue(all(d in RECALL_DOC for d in bench.RECALL_DECOYS))
        middle = RECALL_DOC.index(bench.RECALL_ANSWER) / len(RECALL_DOC)
        self.assertTrue(0.35 < middle < 0.65, middle)
        self.assertTrue(all(c.ok for c in bench.grade_recall("7391-LK")))
        self.assertTrue(all(c.ok for c in bench.grade_recall("`7391-LK`.")))
        self.assertFalse(bench.grade_recall("1187-MR")[0].ok)
        hedged = by_name(bench.grade_recall("It is 7391-LK (was 1187-MR before)."))
        self.assertFalse(hedged["names no near-miss passcode"])
        self.assertFalse(by_name(bench.grade_recall("The passcode is 7391-LK"))["answers with only the passcode"])

    def test_json_strict_grader(self):
        self.assertTrue(all(c.ok for c in bench.grade_json_strict(JSON_GOOD)))
        self.assertFalse(bench.grade_json_strict(f"```json\n{JSON_GOOD}\n```")[0].ok)
        self.assertFalse(bench.grade_json_strict(f"Here you go: {JSON_GOOD}")[0].ok)

        def broken(**changes):
            return by_name(bench.grade_json_strict(json.dumps({**json.loads(JSON_GOOD), **changes})))
        self.assertFalse(broken(extra=1)["exactly the five keys"])
        self.assertFalse(broken(priority="urgent")['priority is "high"'])
        self.assertFalse(broken(tags=["auth", "auth", "outage"])["tags are 3 distinct lowercase words of 3-12 letters"])
        self.assertFalse(broken(tags=["Auth", "timeout", "outage"])["tags are 3 distinct lowercase words of 3-12 letters"])
        self.assertFalse(broken(estimate_hours="8")["estimate_hours is an integer from 1 to 40"])
        self.assertFalse(broken(estimate_hours=True)["estimate_hours is an integer from 1 to 40"])
        self.assertFalse(broken(estimate_hours=41)["estimate_hours is an integer from 1 to 40"])
        self.assertFalse(broken(owner="sam")["owner is null"])
        self.assertFalse(broken(title="Login")["title is 10-40 characters and mentions login"])

    def test_logic_puzzle_has_exactly_one_solution_by_brute_force(self):
        survivors = [o for o in permutations(bench.RUNNERS) if all(ok(o) for _, ok in bench.LOGIC_CLUES)]
        self.assertEqual(survivors, [("Ana", "Dee", "Cal", "Ben", "Eli")])
        self.assertEqual(len(list(permutations(bench.RUNNERS))), 120)
        self.assertTrue(all(c.ok for c in bench.grade_logic(LOGIC_GOOD)))
        swapped = by_name(bench.grade_logic("ORDER: Eli,Ana,Dee,Cal,Ben"))
        self.assertFalse(swapped["satisfies every clue"])
        self.assertFalse(bench.grade_logic("ORDER: Ana,Dee,Cal,Ben")[0].ok)
        self.assertFalse(bench.grade_logic("ORDER: Ana,Dee,Cal,Ben,Ben")[0].ok)

    def test_swift_generic_grader_compiles_and_runs(self):
        with mock.patch.object(bench, "swift_outputs", return_value=(bench.GROUP_EXPECTED, "")):
            self.assertTrue(all(c.ok for c in bench.grade_swift_generic(GROUP_SWIFT)))
        with mock.patch.object(bench, "swift_outputs", return_value=(None, "swiftc not installed")):
            self.assertIsNone(by_name(bench.grade_swift_generic(GROUP_SWIFT))["compiles"])
        plain = by_name(bench.grade_swift_generic("```swift\nfunc groupedCounts(_ items: [String]) -> [(String, Int)] { [] }\n```"))
        self.assertFalse(plain["declared generic over Hashable"])
        if bench.shutil.which("swiftc"):  # real compile and run
            good = by_name(bench.grade_swift_generic(GROUP_SWIFT))
            self.assertTrue(all(v is True for v in good.values()), good)
            wrong = by_name(bench.grade_swift_generic(GROUP_TIE_WRONG))
            self.assertFalse(wrong["counts and tie order correct"])
            broken = by_name(bench.grade_swift_generic("```swift\nfunc groupedCounts<T: Hashable>(_ i: [T]) -> Int { 1 }\n```"))
            self.assertFalse(broken["compiles"] and broken["counts and tie order correct"])

    def test_strict_summary_grader(self):
        self.assertTrue(380 <= len(bench.MEMO.split()) <= 450, len(bench.MEMO.split()))
        self.assertTrue(all(c.ok for c in bench.grade_summary_strict(SUMMARY_GOOD)))
        five = by_name(bench.grade_summary_strict(SUMMARY_GOOD + "\n- One more bullet here"))
        self.assertFalse(five["exactly 4 bullets"])
        copied = SUMMARY_GOOD.replace("Contractor bids close February 28; safety record counts as much as price.",
                                      "Contractor selection closes on February 28 and bidders must show experience.")
        self.assertFalse(by_name(bench.grade_summary_strict(copied))["copies no run of 6 words from the text"])
        invented = by_name(bench.grade_summary_strict(SUMMARY_GOOD.replace("2.4 million", "3.1 million")))
        self.assertFalse(invented["invents no numbers"])
        self.assertFalse(invented["keeps the three required facts"])
        long = by_name(bench.grade_summary_strict(SUMMARY_GOOD.replace("Budget is", "The overall budget for the whole project is")))
        self.assertFalse(long["each bullet is 18 words or fewer"])
        chatty = by_name(bench.grade_summary_strict("Here is the summary:\n" + SUMMARY_GOOD))
        self.assertFalse(chatty["nothing but the bullets"])


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def chat(self, role, prompt, system=""):
        if self.settings.coder == "broken":
            raise OllamaError("down")
        return "loop line that is long enough\n" * 10 if self.settings.coder == "loopy" else "ok"


class RunTests(unittest.TestCase):
    def test_run_scores_every_model_and_task(self):
        task = Task("t", "misc", "p", lambda text: [Check("said ok", text == "ok")])
        rows = bench.run(Settings(), ["good", "loopy", "broken"], [task], client_factory=FakeClient)
        scores = {r["model"]: (r["passed"], r["failed"]) for r in rows}
        self.assertEqual(scores["good"], (2, 0))    # said ok + no loop
        self.assertEqual(scores["loopy"], (0, 2))   # wrong text, repetition loop
        self.assertEqual(scores["broken"], (0, 1))  # model never answered
        table = bench.format_table(rows)
        self.assertIn("good", table)
        self.assertIn("total", table)
        self.assertIn("no repetition loop", table)

    def test_every_role_uses_the_model_under_test(self):
        seen = []
        task = Task("t", "misc", "p", lambda text: [], runner=lambda c: seen.append(
            [c.settings.model(r) for r in ("fast", "coder", "reasoner")]) or "x")
        bench.run(Settings(), ["m"], [task], client_factory=FakeClient)
        self.assertEqual(seen, [["m", "m", "m"]])

    def test_cli_lists_tasks_and_rejects_unknown_ones(self):
        with mock.patch("builtins.print") as out:
            self.assertEqual(main(["bench", "--list"]), 0)
        listed = "\n".join(c[0][0] for c in out.call_args_list)
        self.assertIn("timer_html", listed)
        self.assertIn("pipeline_grounded", listed)
        self.assertEqual(main(["bench", "--tasks", "nope"]), 2)

    def test_heavy_tasks_are_opt_in(self):
        self.assertEqual([t.name for t in bench.TASKS if t.heavy], ["pipeline_grounded"])

    def selected(self, *argv):
        seen = []
        with mock.patch.object(bench, "run", side_effect=lambda s, m, tasks, **kw: seen.extend(t.name for t in tasks) or []), \
                mock.patch.object(bench, "format_table", return_value=""), mock.patch("builtins.print"):
            self.assertEqual(main(["bench", *argv]), 0)
        return seen

    def test_hard_tier_is_opt_in_but_nameable(self):
        default = self.selected()
        self.assertEqual(default, [t.name for t in bench.TASKS if t.tier == "easy" and not t.heavy])
        with_hard = self.selected("--hard")
        self.assertEqual(set(with_hard) - set(default), {t.name for t in bench.TASKS if t.tier == "hard"})
        self.assertNotIn("pipeline_grounded", with_hard)
        self.assertEqual(self.selected("--tasks", "hard_csv"), ["hard_csv"])

    def test_list_shows_the_tier(self):
        with mock.patch("builtins.print") as out:
            main(["bench", "--list"])
        rows = {c[0][0].split()[0]: c[0][0] for c in out.call_args_list}
        self.assertIn("hard", rows["hard_json"])
        self.assertIn("easy", rows["timer_html"])


if __name__ == "__main__":
    unittest.main()
