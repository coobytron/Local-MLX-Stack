"""Deterministic, inspectable routing. No extra model needed to select a model."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Coding intent wins over broad reasoning words such as 'plan' and 'debug'.
CODE = re.compile(
    r"\b(code|coding|implement|implementation|fix|bug|refactor|test|tests|unit test|"
    r"python|javascript|typescript|swift|metal shader|shader|webgl|git|pull request|"
    r"repository|repo|function|api endpoint|component|compile|build script)\b",
    re.IGNORECASE,
)
REASON = re.compile(
    r"\b(architect|architecture|strategy|strategic|plan|planning|design a system|"
    r"tradeoffs?|compare|evaluate|audit|review|reason|diagnose|root cause|risk|"
    r"system design|research plan)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Route:
    role: str
    why: str


def select(prompt: str, override: str = "auto") -> Route:
    if override != "auto":
        if override not in ("fast", "coder", "reasoner"):
            raise ValueError(f"Invalid role: {override}")
        return Route(override, "manual selection")
    if CODE.search(prompt):
        return Route("coder", "coding / repository keywords")
    if REASON.search(prompt):
        return Route("reasoner", "planning / analysis keywords")
    return Route("fast", "general question or extraction")
