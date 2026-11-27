"""ReWOO-style decoupled planner, dependency-level builder, and token profiler.

This module uses only the Python standard library. Tools are supplied by the
caller as a mapping from tool names to callables.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable
import re


@dataclass(frozen=True)
class PlanStep:
    var_name: str
    plan_desc: str
    tool_name: str
    raw_arg: str


_VAR_RE = re.compile(r"#E\d+")
_STEP_RE = re.compile(r"^\s*(#E\d+)\s*=\s*([A-Za-z_]\w*)\[(.*)\]\s*$")


def parse_blueprint(blueprint_text: str) -> list[PlanStep]:
    """Parse lines like '#E1 = Search[weather in Berlin]' into PlanStep objects.

    Lines beginning with 'Plan:' and free-form reasoning lines are accepted.
    """
    steps: list[PlanStep] = []
    current_plan = ""
    for line in blueprint_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("plan:"):
            current_plan = stripped.split(":", 1)[1].strip()
            continue
        match = _STEP_RE.match(line)
        if match:
            var_name, tool_name, raw_arg = match.groups()
            steps.append(PlanStep(var_name, current_plan, tool_name, raw_arg.strip()))
            current_plan = ""
        elif not stripped.startswith("#"):
            # Treat a free-form line between steps as the description for the next step.
            current_plan = stripped
    return steps


def _dependencies(step: PlanStep, known_vars: set[str]) -> set[str]:
    return {token for token in _VAR_RE.findall(step.raw_arg) if token in known_vars}


def build_dependency_levels(steps: Iterable[PlanStep]) -> list[list[PlanStep]]:
    """Return topological levels; steps in a level have no dependencies on each other."""
    steps = list(steps)
    by_var = {step.var_name: step for step in steps}
    if len(by_var) != len(steps):
        raise ValueError("Duplicate step variable names found.")
    known = set(by_var)
    deps = {s.var_name: _dependencies(s, known) for s in steps}

    # Reject references to a later/nonexistent variable only if they look like #E tokens.
    all_vars = set(_VAR_RE.findall("\n".join(s.raw_arg for s in steps)))
    unknown = all_vars - known
    if unknown:
        raise ValueError(f"Unknown variable reference(s): {', '.join(sorted(unknown))}")

    levels: list[list[PlanStep]] = []
    completed: set[str] = set()
    remaining = list(steps)
    while remaining:
        ready = [s for s in remaining if deps[s.var_name] <= completed]
        if not ready:
            cycle = ", ".join(s.var_name for s in remaining)
            raise ValueError(f"Dependency cycle detected among: {cycle}")
        levels.append(ready)
        ready_vars = {s.var_name for s in ready}
        completed.update(ready_vars)
        remaining = [s for s in remaining if s.var_name not in ready_vars]
    return levels


def _substitute(value: str, results: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(0)
        if key not in results:
            raise KeyError(f"No result available for {key}")
        return str(results[key])
    return _VAR_RE.sub(replace, value)


def execute_dag(
    steps: Iterable[PlanStep],
    tools: dict[str, Callable[[str], Any]],
    *,
    max_workers: int | None = None,
    replan_on_error: Callable[[PlanStep, Exception], PlanStep | None] | None = None,
) -> dict[str, Any]:
    """Execute dependency levels. Independent calls within a level run concurrently.

    If a tool raises and replan_on_error is supplied, the callback may return one
    replacement PlanStep. A replacement is attempted once for that failed step.
    """
    steps = list(steps)
    levels = build_dependency_levels(steps)
    results: dict[str, Any] = {}

    def run_one(step: PlanStep) -> tuple[str, Any]:
        if step.tool_name not in tools:
            raise KeyError(f"Tool not registered: {step.tool_name}")
        argument = _substitute(step.raw_arg, results)
        try:
            return step.var_name, tools[step.tool_name](argument)
        except Exception as exc:
            if replan_on_error is None:
                raise
            replacement = replan_on_error(step, exc)
            if replacement is None:
                raise
            if replacement.tool_name not in tools:
                raise KeyError(f"Replanned tool not registered: {replacement.tool_name}")
            repl_arg = _substitute(replacement.raw_arg, results)
            return step.var_name, tools[replacement.tool_name](repl_arg)

    for level in levels:
        # All dependencies for this level are already resolved; results are read-only
        # while its workers run.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, step) for step in level]
            for future in as_completed(futures):
                var_name, value = future.result()
                results[var_name] = value
    return results


def compare_token_cost(
    k_steps: int,
    base_tokens: int,
    step_tokens: int,
    obs_tokens: int,
) -> dict[str, float | int]:
    """Compare cumulative ReAct and ReWOO input-token costs.

    ReAct: sum from t=1..K of base + (t-1)*(step+observation).
    ReWOO: 2*base + K*(step+observation), following the assignment formula.
    """
    if min(k_steps, base_tokens, step_tokens, obs_tokens) < 0:
        raise ValueError("Token counts and number of steps must be non-negative.")
    react = k_steps * base_tokens + (k_steps * (k_steps - 1) // 2) * (step_tokens + obs_tokens)
    rewoo = 2 * base_tokens + k_steps * (step_tokens + obs_tokens)
    savings = ((react - rewoo) / react * 100) if react else 0.0
    speedup = (react / rewoo) if rewoo else 0.0
    return {
        "react_tokens": react,
        "rewoo_tokens": rewoo,
        "savings_pct": round(savings, 2),
        "speedup_ratio": round(speedup, 2),
    }


if __name__ == "__main__":
    demo = """Plan: Find a topic
#E1 = Search[black holes]
Plan: Summarize the result from the first step
#E2 = Summarize[#E1]
"""
    parsed = parse_blueprint(demo)
    print("Parsed:", parsed)
    print("Levels:", [[s.var_name for s in level] for level in build_dependency_levels(parsed)])
    print("Execution:", execute_dag(parsed, {
        "Search": lambda q: f"Results for {q}",
        "Summarize": lambda q: f"Summary of {q}",
    }))
    print("Token profile:", compare_token_cost(5, 100, 30, 20))
