# Local MLX Stack, simplified

*Updated 2026-11-20. Everything below was measured on this M5 Max (48GB). Nothing has been run on the two M1 Max Macs yet.*

## The one-minute version

- **What it is:** a small app that runs AI models **on your Mac, offline**, picks the right one for each job, and can edit code through OpenCode.
- **Use it:** double-click `start.command`, open http://127.0.0.1:8788.
- **The #1 lesson:** OpenCode loops forever when the model's context is too small. The **32k copies of the models fix it** (`stack setup` makes one).
- **Best everyday model here:** `qwen3-coder-32k`. Same quality as gemma on easy work, about 5 times faster.

## How we got here (the story in 8 steps)

| # | What we did | What we learned |
|---|---|---|
| 1 | Read the stack: a keyword router sends each prompt to one of 3 models | No AI picks the model, so routing is free and predictable (12/12 on typical prompts) |
| 2 | Built a web UI: Status, Ask, Pipeline, OpenCode, Research | Easier to understand than the command line; copy buttons help |
| 3 | Hit install snags (iCloud hid files, `:latest` names, pasted paths) | Keep the Python env **outside iCloud** (`~/.venvs/mlx-stack`) |
| 4 | OpenCode looped 25 times on a finished job | Cause: **small context**. The 32k model finished cleanly |
| 5 | Built `stack bench`: 19 chat tasks graded by plain code | Both models tie on easy tasks; the hard tier separates them |
| 6 | Built `stack bench --agentic`: OpenCode edit jobs on throwaway copies | Proved the loop was context: gemma looped at default, not at 32k |
| 7 | Added `stack setup` and `--repeat N` | 32k is now the default; repeats expose lucky or unlucky runs |
| 8 | Had agents build each step, then other agents review it | Reviewers caught a real bug (streamed replies broke `stack setup`) |

## Results at a glance

| Test | qwen3-coder-32k | gemma4:12b-mlx |
|---|---|---|
| 18 chat tasks (easy + hard) | **100/104 in 94s** | 95/104 in 522s |
| Biggest chat gap | semver parsing 7/10 | semver parsing 2/10; invented a function that doesn't exist |
| OpenCode edit jobs | add flag ✅, fix bug ✅, 5-file rename ✅ (once stalled), translate ⚠️ | at 32k: add flag ✅, others partial |

**Verdict:** use the 30b for OpenCode and anything multi-file. Gemma is fine for quick single-file or chat jobs, but slower and shakier.

## Rules of thumb

1. **One file, one task, one run** for OpenCode. Split big jobs.
2. **Never trust the model's summary.** Open the file or run the check.
3. **Always use the 32k model for OpenCode.** `stack setup` once, then `stack opencode` uses it.
4. **Give the Pipeline a `--project` folder**, or it invents files.
5. **Run benchmarks 3 times** (`--repeat 3`): single runs mislead.

## Cheat sheet

```bash
~/.venvs/mlx-stack/bin/stack doctor                       # is everything installed?
~/.venvs/mlx-stack/bin/stack setup                        # make the 32k OpenCode model (once)
~/.venvs/mlx-stack/bin/stack bench --hard --repeat 3 --models qwen3-coder-32k,gemma4:12b-mlx
~/.venvs/mlx-stack/bin/stack bench --agentic --repeat 3 --models qwen3-coder-32k
```

## Offline or not

| Part | Offline? |
|---|---|
| Ask, Pipeline, OpenCode edits, benchmarks | ✅ yes |
| Research tab (crawls a website), OpenCode web fetch | ❌ needs internet |
| Search your notes by meaning (`nomic-embed-text` is installed) | ✅ possible, **nothing uses it yet** |

## M5 Max vs M1 Max: what we know

- ✅ **M5 Max (this Mac):** everything above, measured here.
- ❓ **M1 Max (two spare Macs):** untested. Expect **slower** answers, and **the 32k models need more memory**. Check how much each has (32GB or 64GB) before pulling the 30b.
- **Simplest way to use them:** run Ollama on an M1 Max and point this stack at it with an SSH tunnel (`ssh -L 11435:127.0.0.1:11434 <m1>`, then `OLLAMA_HOST=http://127.0.0.1:11435`). No code change.
- **A fair comparison:** run the same `stack bench --repeat 3` on each Mac and compare the tables.

## Still unknown

- Memory use and exact speed per model (we only have total seconds).
- Whether the Research crawler works well on real sites.
- Whether any small model can drive OpenCode on multi-file jobs reliably.

## Gotchas we already fixed

iCloud hiding files (env moved out) · `:latest` model names · quoted pasted paths · OpenCode using the wrong folder (`$PWD`) · `stack setup` failing on streamed replies · bad env values crashing with a traceback.
