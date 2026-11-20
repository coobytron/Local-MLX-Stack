# Local MLX Stack: what we found, what it can do offline

*Written 2026-11-19 from our hands-on sessions. "Verified" means we saw it happen. No benchmark numbers exist yet.*

## 1. This computer, at a glance

| Item | Value |
|---|---|
| Machine | M5 Max Mac, 48GB unified memory, macOS 27, Apple Silicon (arm64) |
| Tools present | Ollama (running), OpenCode, Swift 6.4.2, Node, Python 3.11, M5Unified/M5GFX Arduino libraries |
| Tool missing | Metal compiler toolchain (`xcodebuild -downloadComponent MetalToolchain` to add) |
| Models installed | `gemma4:12b`, `gemma4:12b-mlx`, `mistral-small3.1`, `qwen3-coder:30b`, `qwen3.5:35b-mlx`, `nomic-embed-text`, plus `qwen3-coder-32k` (our 32k-context copy of the 30b) |
| Run it | double-click `start.command`, open http://127.0.0.1:8788 |
| Python env | `~/.venvs/mlx-stack` (outside iCloud on purpose) |

## 2. What we can do offline

| Capability | Offline? | Status |
|---|---|---|
| Ask a question (router picks 1 model) | ✅ | Works |
| Pipeline: brief → plan → draft → review on a real project folder | ✅ | Works; text only, never edits files |
| Web UI: Status, Ask, Pipeline, OpenCode command builder, Copy buttons | ✅ | Works |
| OpenCode edits real files with a local model | ✅ | Works with the 32k model, one file at a time |
| `stack bench` (11 tasks, plain-code grading) | ✅ | Built and unit-tested; **never run on real models yet** |
| Research tab (crawl a site, one summary per page) | ❌ needs internet | Built and tested on a fake site; **not tried on a real site** |
| OpenCode web fetch | ❌ needs internet | Exists; the model guessed wrong URLs when left alone |
| Search by meaning across your notes (`nomic-embed-text`) | ✅ possible | Model installed, **nothing uses it yet** |

## 3. Key findings

1. ✅ **The "loop" was OpenCode's auto-summary repeating, not the model going crazy.** With the small default context it summarized itself 25 times after finishing a simple job. With the 32k model the same kind of job finished and stopped. (We never ran `ollama ps`, so the exact cause is an informed guess.)
2. ⚠️ **Big multi-file jobs still stall.** Translating a whole app (15 files) made it re-read the same files and summarize without editing anything. **One file per run worked**: about 60 strings translated.
3. ⚠️ **The model's own report is unreliable.** After the good run it said "5 strings changed" and listed 8, and claimed it left strings that weren't in the file. It also missed 6 strings. Check the file, not the summary.
4. ✅ **Small model is competitive on simple jobs.** On the single-file timer, `gemma4:12b-mlx` had fewer bugs than `qwen3-coder:30b`. One test only.
5. ⚠️ **Models guess when they can't look.** Given web access, OpenCode invented a GitHub owner and got two 404s. Models invent file names too, which is why the pipeline needs `--project`.
6. ✅ **Chat answers and agent work are different skills.** Passing a chat test doesn't mean OpenCode will finish a multi-step edit.

### First benchmark run (2026-11-19, 10 easy chat tasks)

| | qwen3-coder-32k | gemma4:12b-mlx |
|---|---|---|
| Score | 52/52 | 52/52 |
| Total time | 57s | 286s |

✅ Quality tie, ✅ the 30b is ~5x faster. ⚠️ Too easy to separate the models, so we added a hard tier and an agentic bench.

### Hard tier (chat, 8 extra tasks)

| | qwen3-coder-32k | gemma4:12b-mlx |
|---|---|---|
| All 18 tasks | 100/104 in 94s | 95/104 in 522s |

Biggest gap: `hard_semver` (30b 7/10, gemma 2/10). Gemma invented a function (`getBattery`) that doesn't exist in M5Unified.

### Agentic bench (OpenCode edits throwaway copies; 1 run each, so treat as a signal)

| Task | qwen3-coder-32k | gemma 16k (default) | gemma 32k |
|---|---|---|---|
| add_flag | ✅ 6/6 | ❌ looped, no edit | ✅ 7/7 |
| fix_bug | ✅ 6/6 | ⚠️ looped | ❌ no edit |
| translate_file | ⚠️ 8/10 (also translated comments) | ❌ looped | ⚠️ 9/10 (broke indentation) |
| rename_multi (5 files) | ✅ 8/8 twice, 1 stall | ❌ looped | ⚠️ 5/8 (3 files missed) |

✅ **Context size caused the summary loop:** gemma looped on all 4 tasks at the default context and none at 32k. ⚠️ **Small model is shaky on multi-file edits.** ⚠️ **qwen3-coder sometimes prints a tool call as text** (`<function=grep>`) and stops (1 of ~4 runs).

## 4. Setup gotchas we hit (all fixed)

| Problem | Fix |
|---|---|
| iCloud hides the venv's `.pth` files, so `stack` can't find its code | Venv moved to `~/.venvs/mlx-stack` |
| Status said a model was "missing" | Ollama names untagged models `name:latest`; the check now matches both |
| Pasted paths with quotes or `\ ` failed | The UI now cleans them |
| `stack opencode` ignored the config's default model | Set `STACK_CODER_MODEL=qwen3-coder-32k` first |
| OpenCode needs a bigger context | `ollama create qwen3-coder-32k` from a Modelfile with `num_ctx 32768` |

## 5. Rules of thumb

- **One file, one task, one run** for OpenCode. Bigger jobs: split them.
- **Always check the result yourself** (grep, open the file); don't trust the summary.
- **Use `--project`** for the Pipeline so it reads real files.
- **A classic ESP32 can't do USB HID**; an S2 or S3 can.

## 6. Not known yet

- Real speed and memory numbers (no timing data collected).
- Which model is best at which job. Run `stack bench` to get that.
- Whether any small model can drive OpenCode reliably.
- Whether the Research crawler behaves on real websites.

## 7. Next steps, in order

1. Run `stack bench --models qwen3-coder-32k,gemma4:12b-mlx --output ~/bench.jsonl` and read the table.
2. Add an **agentic bench** (OpenCode on scratch copies, graded by files changed and loop detection).
3. Try Research on one small public site.
4. Parked: two spare M1 Max Macs as extra Ollama hosts (SSH tunnel, no code change needed).
