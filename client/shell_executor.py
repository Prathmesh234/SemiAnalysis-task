"""shell_executor.py — Run `claude --print "<prompt>"` as a subprocess with env vars."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Captures stdout, stderr, and return code from a claude CLI invocation."""

    stdout: str
    stderr: str
    returncode: int


async def run_claude_print(
    prompt: str,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 600,
) -> ExecutionResult:
    """Invoke ``claude --print "<prompt>"`` and return captured output.

    Parameters
    ----------
    prompt:
        The prompt string passed to claude CLI.
    env_overrides:
        Extra environment variables merged on top of os.environ.
    timeout:
        Max seconds before the subprocess is killed.
    """
    env = {**os.environ, **(env_overrides or {})}

    proc = await asyncio.create_subprocess_exec(
        "claude", "--print", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return ExecutionResult(stdout="", stderr="TIMEOUT", returncode=-1)

    return ExecutionResult(
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        returncode=proc.returncode or 0,
    )
