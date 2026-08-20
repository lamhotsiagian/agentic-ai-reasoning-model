"""python -m app.reasoning.eval [suite]: run the task suite and print the
decomposed report."""
from __future__ import annotations

import asyncio
import json
import sys

from app.reasoning.eval.harness import run_suite, run_tool_fixtures


async def main() -> None:
    suite = sys.argv[1] if len(sys.argv) > 1 else "default"
    tools = await run_tool_fixtures()
    tasks = await run_suite(suite)
    print(json.dumps({"tools": tools, "tasks": tasks}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
