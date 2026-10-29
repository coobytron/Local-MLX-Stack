# Local MLX Stack — Agent instructions

This project follows [Agent Cody Banks](https://github.com/coobytron/Agent-Cody-Banks) rather than maintaining a second roster of agents.

- **One accountable primary by default.** Do not automatically create a multi-agent team for a task.
- Only **researcher**, **implementer**, and **reviewer** are delegable workers, and only for exactly one named trigger: `parallel`, `isolation`, `tools`, or `review`. Maximum two workers concurrently and no nested delegation.
- Capability names (AI Specialist, Architect, Creative Technologist, Swift Expert, etc.) are **checklists**, not independent agents. Find the relevant file in `checklists/INDEX.md` in the ACB checkout.
- Respect explicit user instructions, approved project briefs and acceptance criteria, then repository tests and code; distinguish speculation from observed results.
- Check real tool access before claiming code changes, test results, published artifacts, or image inspection.
- Worker tool restrictions and delegation hooks are enforced inside Claude Code with the ACB plugin. **This local router does not implement or guarantee those tool boundaries.** OpenCode permissions must be configured separately.
- Never mutate a target repository during analysis-only `stack ask` or `stack pipeline` calls. Only `stack opencode --execute` can launch a separate tool-capable coding harness.
- Use `python3 -m unittest discover -s tests -v` to validate changes in this repository.

Source of truth: `https://github.com/coobytron/Agent-Cody-Banks/blob/main/AGENTS.md` and `docs/ORCHESTRATION.md`.
