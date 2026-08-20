"""Tool layer (Chapter 5).

The model is an unreliable client calling your API, and you cannot patch the
client. Your only levers are the contract you expose and the way you handle
what comes back.
"""
from app.reasoning.tools.contract import (ToolEmpty, ToolError, ToolOk,
                                          ToolResult)
from app.reasoning.tools.router import CircuitBreaker, ToolRouter, ToolSpec

__all__ = ["ToolOk", "ToolEmpty", "ToolError", "ToolResult",
           "ToolRouter", "ToolSpec", "CircuitBreaker"]
