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
    verbose: bool = True,
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
    verbose:
        If True, print stdout/stderr in real time.
    """
    env = {**os.environ, **(env_overrides or {})}

    # Use the SGLang model name so the proxy forwards with the correct model field
    model = env.get("MODEL_PATH", "zai-org/GLM-4.5-Air-FP8")

    if verbose:
        print(f"  ▸ claude --print --model {model} (prompt={len(prompt)} chars)...", flush=True)

    proc = await asyncio.create_subprocess_exec(
        "claude", "--print", "--model", model, prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        if verbose:
            print("  ✗ TIMEOUT", flush=True)
        return ExecutionResult(stdout="", stderr="TIMEOUT", returncode=-1)

    stdout_str = stdout.decode()
    stderr_str = stderr.decode()

    if verbose:
        rc = proc.returncode or 0
        icon = "✓" if rc == 0 else "✗"
        print(f"  {icon} exit={rc} stdout={len(stdout_str)} chars", flush=True)
        if stderr_str.strip():
            # Print first few lines of stderr for debugging
            for line in stderr_str.strip().split("\n")[:5]:
                print(f"    stderr: {line}", flush=True)

    return ExecutionResult(
        stdout=stdout_str,
        stderr=stderr_str,
        returncode=proc.returncode or 0,
    )
