"""health_check.py — Poll localhost:8000/health every 5s, timeout after 5min."""

import asyncio
import sys

import httpx

HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL_S = 5
TIMEOUT_S = 300


async def wait_for_healthy(
    url: str = HEALTH_URL,
    poll_interval: float = POLL_INTERVAL_S,
    timeout: float = TIMEOUT_S,
) -> bool:
    """Block until the server at *url* returns 200 or *timeout* expires.

    Returns True if healthy, False on timeout.
    """
    elapsed = 0.0
    async with httpx.AsyncClient() as client:
        while elapsed < timeout:
            try:
                resp = await client.get(url, timeout=5)
                if resp.status_code == 200:
                    print(f"Server healthy after {elapsed:.0f}s")
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
    print(f"Server not healthy after {timeout:.0f}s — giving up")
    return False


if __name__ == "__main__":
    ok = asyncio.run(wait_for_healthy())
    sys.exit(0 if ok else 1)
