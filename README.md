# Local MLX Stack

A **local-first, dependency-light Ollama model router** for an Apple Silicon Mac with 48GB unified memory. It uses your installed models as a coordinated stack without trying to load three large models at the same time. It has an optional OpenCode handoff for real repository changes and reads Agent Cody Banks checklists from a local checkout when available.

**Models in the starter profile**

| Role | Ollama tag | Use |
| --- | --- | --- |
| Fast | `gemma4:12b` | Extraction, summaries, short prompts |
| Coder | `qwen3-coder:30b` | Code suggestions and OpenCode coding sessions |
| Reasoner | `qwen3.5:35b-mlx` | Architecture, planning, independent critique |

Every default `stack ask` request runs **one** model. A deterministic router selects it; use `--role` to override. `stack pipeline` is an explicit, sequential multi-model **text-only** workflow. Ollama receives `keep_alive=0` so each model is unloaded after its turn. A per-process lock serializes generation requests through this service; the stack does not control other Ollama clients.

## Quick start (macOS)

```bash
# Your third model may still be downloading — confirm when it finishes:
ollama list

# Get the project
cd ~/Dev
git clone https://github.com/coobytron/Local-MLX-Stack.git
cd Local-MLX-Stack

# No runtime Python dependencies beyond the standard library
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Check installed model names and Ollama availability
stack doctor

# Try the single-model router
stack route "Write a Metal shader"
stack ask "Explain why my SDXL ControlNet pipeline is slow"
stack ask --role reasoner "Design a local asset-indexing architecture"
```

If Ollama isn't running, launch the installed Ollama application or `ollama serve` (not both at once). Missing models can be pulled with:

```bash
ollama pull gemma4:12b
ollama pull qwen3-coder:30b
ollama pull qwen3.5:35b-mlx
```

The listed tags are initial defaults, not assumptions about what's installed. `stack doctor` checks against `/api/tags` and reports missing models. Use `STACK_FAST_MODEL`, `STACK_CODER_MODEL`, or `STACK_REASONER_MODEL` to change them.

## Using the models together

The opt-in pipeline goes through brief extraction → implementation plan → code proposal → (when `--trigger review`) independent model review. It **does not** edit project files, execute shell commands, or claim it ran tests.

```bash
stack pipeline "Design automatic ControlNet selection based on image references" \
  --trigger review --output ./review.json
```

The single named trigger follows [Agent Cody Banks](https://github.com/coobytron/Agent-Cody-Banks). Other supported triggers are `parallel`, `isolation`, and `tools`; in that case the text pipeline stops after the draft. These triggers label intent rather than creating fully permissioned workers. ACB's Claude Code enforcement hooks are **not** active in this tool.

## Ground a pipeline in an actual project (recommended)

The pipeline is **text-only**: without project files it can hallucinate a new
app instead of extending your existing code. Use `--project` to provide a bounded,
read-only snapshot of the current repository. Files are inspected locally;
common generated directories, secrets/config, and executable agent instructions
are excluded. Source snippets are never executed. Only a subset of files is
included, so missing behavior must remain unverified until inspected.

For your existing Draw Things Control app (React/Vite + FastAPI with optional
Ollama/LM Studio assistance and HTTP/gRPC Draw Things adapters):

```bash
stack pipeline "Extend existing Draw Things Control with an opt-in AI controller; preserve current local-model discovery, capability preflight, HTTP/gRPC transports and user approval" \
  --project ~/Dev/draw-things-control \
  --no-acb \
  --trigger review \
  --output ./draw-things-controller-proposal.json
```

Set `--project` to your actual checkout location. `--no-acb` skips optional
external agent guidance **for this invocation only**; it does not modify that
other repository or its plugin. `--trigger review` adds a text review, not a
worker with tool permissions. The JSON includes `project` and `context_files`
so you can inspect exactly which files informed the model.

The project snapshot is intentionally restricted and **does not replace**
reading the full source before implementing changes. Keep the existing typed
Draw Things contracts, loopback-only security, and explicit user approval.

## OpenCode: real code edits

This repo includes an `opencode.json` using the Ollama OpenAI-compatible endpoint. The wrapper forwards that configuration to OpenCode with `OPENCODE_CONFIG`, works in your target project directory, and defaults to a **dry run**:

```bash
stack opencode "Add tests for the history restore feature" \
  --project ~/Dev/draw-things-control

# When you are ready for OpenCode to run and request its usual tool permissions:
stack opencode "Add tests for the history restore feature" \
  --project ~/Dev/draw-things-control --execute
```

Install [OpenCode](https://opencode.ai/) separately. It may ask you to approve file edits or commands; the stack intentionally does **not** auto-approve them. Run `opencode run --help` to verify your installed OpenCode version. OpenCode configuration has evolved between releases; if your version already autodiscovers Ollama, that is another supported route. The wrapper never invokes `opencode` unless you pass `--execute`.

The target project can override models and permissions with its own OpenCode configuration. To use OpenCode interactively rather than with a single task, run `opencode` from the target repository and select `ollama/qwen3-coder:30b`.

## API for Draw Things or your other apps

```bash
stack serve --port 8788

curl -s http://127.0.0.1:8788/health
curl -s http://127.0.0.1:8788/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Implement a Swift Metal image filter"}'
curl -s http://127.0.0.1:8788/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Summarize this image processing requirement","role":"auto"}'
```

- `GET /health` checks Ollama readiness and installed models.
- `POST /route` returns a model selection without calling inference.
- `POST /chat` routes and responds using the local Ollama `/api/chat` endpoint.
- JSON request bodies must contain `prompt`, and may contain `role` (`auto`, `fast`, `coder`, `reasoner`).
- The HTTP server binds exclusively to `127.0.0.1`; it doesn't support browser cross-origin requests or tool execution. Use an application backend to call it. Don't expose it directly to a public network.

## Agent Cody Banks integration

The repository's existing architecture is **one primary agent, at most two bounded workers, no nested delegation**, with executable `researcher`, `implementer`, and `reviewer` roles. Other skill names are capability **checklists**, not workers. This project doesn't duplicate or alter ACB's roster.

```bash
cd ~/Dev
git clone https://github.com/coobytron/Agent-Cody-Banks.git
# Then, from Local-MLX-Stack:
export ACB_ROOT="$HOME/Dev/Agent-Cody-Banks"
stack doctor
```

When that local checkout is available, the router reads its `AGENTS.md` and a task-relevant capability checklist as **guidance**. It doesn't run its `.claude/agents` workers or enforce Claude Code's delegation hook. To use the *actual* enforced ACB roles, install its Claude Code plugin as described in its [README](https://github.com/coobytron/Agent-Cody-Banks/blob/main/README.md). The `--acb-root` option lets you point at another checkout; its location is not uploaded.

## Tune for 48GB RAM

```bash
export STACK_NUM_CTX=16384       # start here; try 32768 if headroom permits
export STACK_KEEP_ALIVE=0        # unload after each response; minimizes resident weights
export STACK_TIMEOUT=600        # seconds per model request
export OLLAMA_HOST=http://127.0.0.1:11434
```

Large-context and concurrent model use can exhaust memory. This stack serializes its own requests and deliberately unloads between calls. Other apps using Ollama independently can still load extra models. If you prefer a small resident fast model later, change `STACK_FAST_MODEL` and `STACK_KEEP_ALIVE`, then observe Activity Monitor memory pressure.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests use a fake local Ollama HTTP server and check routing, request payloads, error handling and no accidental OpenCode execution. They do not benchmark the real M5 Max or verify OpenCode integration on your Mac.

## Project boundary

This is a **working MVP**, not a full multi-agent permission sandbox, RAG/memory server, or visual dashboard. Future additions can include project-scoped search, structured task traces and tool permissions. Model replies are untrusted suggestions until a coding harness or human verifies them.
