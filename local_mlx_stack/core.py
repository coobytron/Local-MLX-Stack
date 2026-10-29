"""Single-primary task execution plus explicitly opted-in staged pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from .acb import guidance, TRIGGERS
from .ollama import OllamaClient
from .routing import select


@dataclass
class Answer:
    role: str
    model: str
    reason: str
    response: str

    def as_dict(self) -> dict:
        return asdict(self)


class Stack:
    def __init__(self, client: OllamaClient, acb_root: Path | None = None):
        self.client = client
        self.acb_root = acb_root

    def ask(self, prompt: str, role: str = "auto") -> Answer:
        route = select(prompt, role)
        reply = self.client.chat(route.role, prompt, guidance(prompt, route.role, self.acb_root))
        return Answer(route.role, self.client.settings.model(route.role), route.why, reply)

    def pipeline(self, task: str, trigger: str) -> dict:
        # This is sequential text generation, not a delegation to tools or a repo mutation.
        if trigger not in TRIGGERS:
            raise ValueError("A single named ACB trigger is required: parallel, isolation, tools, review")
        brief = self.client.chat(
            "fast", f"Identify the objective, constraints, acceptance criteria and unanswered questions.\n\n{task}",
            guidance(task, "fast", self.acb_root)
        )
        plan = self.client.chat(
            "reasoner", f"Design a concrete, testable implementation plan for this brief. Do not claim execution.\n\n{brief}",
            guidance(task, "reasoner", self.acb_root)
        )
        draft = self.client.chat(
            "coder", f"Propose code or a precise implementation recipe for this plan. "
            f"You cannot read files or run tests here.\n\nOriginal task:\n{task}\n\nPlan:\n{plan}",
            guidance(task, "coder", self.acb_root)
        )
        output = {"trigger": trigger, "brief": brief, "plan": plan, "draft": draft,
                  "models": {role: self.client.settings.model(role) for role in ("fast", "coder", "reasoner")}}
        if trigger == "review":
            output["review"] = self.client.chat(
                "reasoner", f"Trigger: review\nProvide an independent critical review of this proposed "
                "implementation. No tools have been run. Identify specific gaps and tests needed.\n\n"
                f"Task:\n{task}\n\nProposal:\n{draft}",
                guidance(task, "reasoner", self.acb_root)
            )
        return output
