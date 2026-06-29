---
title: Advanced AI Research Agent
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: true
short_description: ReAct agent with live web search, math, and Wikipedia — chat UI
---

# 🤖 Advanced AI Research Agent

A **production-grade ReAct agent** with a full chat UI. Powered by Qwen3-32B via Groq.

Ask it anything — it searches the web in real time, does exact math, and explains concepts using Wikipedia. Tool calls appear live in the chat as the agent thinks.

## Tools

| Tool | Description |
|---|---|
| 🔍 Web Search | Live DuckDuckGo — current prices, news, job info |
| 🧮 Calculator | Exact math: `5600 * 12`, `sqrt(144)`, compound interest |
| 📅 Date & Time | Current date/time |
| 📖 Wikipedia | Definitions, history, technical concepts |

## How to Use

1. Enter your **GROQ API key** in the sidebar (free at [console.groq.com](https://console.groq.com))
2. Type any question and press **Send**
3. Watch the agent reason — tool calls appear in real time
4. Results are sourced, calculated, and structured

## Architecture

```
User question
     ↓
Qwen3-32B (Groq) — decides: use tool or answer?
     ↓ tool call
Execute tool (web_search / calculate / get_date / wikipedia)
     ↓ result fed back
LLM reasons again → repeat until final answer
```

**Pattern:** ReAct (Reason + Act)  
**Model:** Qwen3-32B via Groq (~800 tok/s, free tier)  
**Framework:** Gradio 6 · Python

## Stack

`gradio` · `groq` · `ddgs` · `requests`
