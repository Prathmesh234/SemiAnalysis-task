"""turn_manager.py — Multi-turn loop: run_turn() → append output → rebuild prompt → repeat."""

from __future__ import annotations

from dataclasses import dataclass, field

from client.shell_executor import run_claude_print, ExecutionResult


@dataclass
class TurnResult:
    """Accumulated results across all turns for a single task."""

    turns: list[ExecutionResult] = field(default_factory=list)
    final_output: str = ""


async def run_multi_turn(
    initial_prompt: str,
    max_turns: int = 5,
    env_overrides: dict[str, str] | None = None,
) -> TurnResult:
    """Execute up to *max_turns* of claude CLI calls, feeding each output back.

    Each turn appends the previous assistant output to the prompt so the agent
    can iterate on its own work.
    """
    result = TurnResult()
    prompt = initial_prompt

    for _ in range(max_turns):
        exec_result = await run_claude_print(prompt, env_overrides=env_overrides)
        result.turns.append(exec_result)

        if exec_result.returncode != 0:
            break

        result.final_output = exec_result.stdout

        # Rebuild prompt with prior output for next turn
        prompt = f"{initial_prompt}\n\n# Previous output:\n{exec_result.stdout}"

    return result
