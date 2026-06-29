"""
Streaming ReAct Agent
=====================
Generator-based agent that yields step events as it reasons:
  {"type": "tool_start",  "tool": ..., "args": ..., "iteration": ...}
  {"type": "tool_result", "tool": ..., "result": ..., "iteration": ...}
  {"type": "final",       "answer": ..., "steps": [...], "iterations": ..., "time_ms": ...}

Designed so the Gradio UI can stream tool calls in real-time.
"""

import json
import os
import re
import time

from groq import Groq
from tools import TOOL_SCHEMAS, TOOL_META, execute_tool

MODEL = "qwen/qwen3-32b"

SYSTEM_PROMPT = """You are an expert AI research assistant with real-time access to powerful tools.

Your approach:
1. Break complex questions into sub-tasks
2. Use tools strategically — web_search for current facts, calculate for any math, wikipedia for concepts
3. Cross-reference results when accuracy is critical
4. Give clear, structured, well-sourced answers

Response format:
- Use markdown: **bold** for key facts, bullet points for lists, headers for sections
- Always show numbers from calculations (never estimate math)
- Cite sources: mention where facts came from
- Be concise but complete — the user wants actionable information

You have 4 tools:
- web_search: live internet search (DuckDuckGo) — best for current data
- calculate: exact math evaluation — use for ALL calculations
- get_current_date: today's date/time
- wikipedia_search: definitions, history, technical concepts
"""


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 prepends."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _format_arg_preview(tool: str, args: dict) -> str:
    """Short human-readable preview of tool arguments."""
    if tool == "web_search":
        return f'"{args.get("query", "")}"'
    if tool == "calculate":
        return f'`{args.get("expression", "")}`'
    if tool == "wikipedia_search":
        return f'"{args.get("topic", "")}"'
    if tool == "get_current_date":
        return ""
    return str(args)[:80]


def agent_stream(user_message: str, api_key: str, max_iterations: int = 8):
    """
    Generator: runs the ReAct loop and yields step events.
    Each call to this function is stateless — fresh context window.
    """
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    start_time = time.time()
    steps = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=2048,
        )

        message = response.choices[0].message

        # ── LLM wants to call tools ──────────────────────────────────────
        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                arg_preview = _format_arg_preview(tool_name, tool_args)

                yield {
                    "type": "tool_start",
                    "tool": tool_name,
                    "args": tool_args,
                    "arg_preview": arg_preview,
                    "iteration": iteration,
                }

                result = execute_tool(tool_name, tool_args)

                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "result": result,
                    "iteration": iteration,
                }

                steps.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_length": len(result),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # ── LLM gives final answer ───────────────────────────────────────
        else:
            final = _strip_thinking(message.content or "")
            elapsed_ms = (time.time() - start_time) * 1000

            yield {
                "type": "final",
                "answer": final,
                "steps": steps,
                "iterations": iteration,
                "time_ms": elapsed_ms,
            }
            return

    # Exceeded max_iterations
    elapsed_ms = (time.time() - start_time) * 1000
    yield {
        "type": "final",
        "answer": "I reached the maximum reasoning steps. Here is what I found up to this point.",
        "steps": steps,
        "iterations": iteration,
        "time_ms": elapsed_ms,
    }
