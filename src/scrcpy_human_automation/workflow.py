from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


Step = Callable[["AutomationContext"], None]


@dataclass
class AutomationContext:
    state: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.state:
            raise KeyError(f"Missing required context value: {key}")
        return self.state[key]


class AutomationFlow:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[tuple[str, Step]] = []

    def add_step(self, name: str, step: Step) -> "AutomationFlow":
        self.steps.append((name, step))
        return self

    def run(self, context: AutomationContext | None = None) -> AutomationContext:
        ctx = context or AutomationContext()
        for step_name, step in self.steps:
            ctx.set("last_step", step_name)
            step(ctx)
        return ctx
