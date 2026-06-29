"""
AI Research Agent — Production Gradio UI (Gradio 6.x)
========================================================
Full chat interface with real-time tool call display,
multi-turn history, stats panel, and example questions.

Deploy: HuggingFace Spaces · Gradio SDK · CPU Basic (free)
Secret: Set GROQ_API_KEY in Space Settings → Variables and secrets
"""

import os
import gradio as gr
from agent import agent_stream
from tools import TOOL_META

# ── Constants ───────────────────────────────────────────────────────────────

ENV_KEY = os.environ.get("GROQ_API_KEY", "")

EXAMPLES = [
    "What is the minimum salary for a Singapore Employment Pass in 2026, and how much is that in USD per year?",
    "Explain how Retrieval-Augmented Generation (RAG) works and what problems it solves for LLMs.",
    "If I invest $10,000 at 8% compound interest for 10 years, how much will I have?",
    "What is the difference between LoRA and full fine-tuning of LLMs?",
    "What are the top AI companies hiring in Singapore right now?",
    "What is today's date, and how many days are left until the end of 2026?",
    "Summarize the current state of AI regulation in the EU and Singapore.",
    "What is the Transformer architecture? Use Wikipedia to explain it.",
]

TOOL_CARDS = [
    ("🔍", "Web Search",  "Live DuckDuckGo search — current news, prices, job info, events."),
    ("🧮", "Calculator",  "Evaluates math exactly: arithmetic, sqrt, compound interest, etc."),
    ("📅", "Date & Time", "Returns current date/time — for time-sensitive queries."),
    ("🌐", "Fetch URL",   "Reads any public URL: GitHub profiles, APIs, news articles, docs."),
    ("📖", "Wikipedia",   "Looks up definitions, history, and technical concepts."),
]

CSS = """
/* Layout */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* Header gradient card */
.header-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #1e40af 100%);
    border-radius: 16px;
    padding: 28px 36px 22px 36px;
    margin-bottom: 18px;
    color: white;
}
.header-card h1 { color: white !important; font-size: 2rem; margin: 0 0 6px 0; }
.header-card p  { color: #c7d2fe !important; margin: 0; font-size: 0.95rem; }

/* Tool call blockquotes inside the chat */
blockquote {
    border-left: 3px solid #6366f1 !important;
    background: #f5f3ff !important;
    margin: 4px 0 !important;
    padding: 6px 12px !important;
    border-radius: 0 8px 8px 0 !important;
    font-size: 0.87rem !important;
}

/* Sidebar tool card */
.tool-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.tool-card:hover { border-color: #6366f1; background: #faf5ff; }

/* Stats panel */
.stats-panel {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 14px;
}

/* Key panel */
.key-panel {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
}

/* Subtle input row */
.input-area { margin-top: 8px; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

HEADER_HTML = """
<div class="header-card">
  <h1>🤖 AI Research Agent</h1>
  <p>Powered by <strong>Qwen3-32B via Groq</strong> · Searches the web in real-time · Does exact math · Explains any concept</p>
</div>
"""

PLACEHOLDER = """
<div style="text-align:center; color:#9ca3af; padding:60px 0;">
  <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
  <div style="font-size:1.1rem; font-weight:600; margin-bottom:6px;">Ask me anything</div>
  <div style="font-size:0.9rem;">I'll search the web, calculate, and explain concepts in real time</div>
</div>
"""


# ── Helper functions ────────────────────────────────────────────────────────

def _result_preview(tool: str, result: str) -> str:
    if tool == "calculate":
        return result
    if tool == "get_current_date":
        return result
    if tool == "web_search":
        count = result.count("[") if "[1]" in result else "several"
        return f"Retrieved {count} results"
    if tool == "wikipedia_search":
        lines = result.split("\n")
        title = lines[0].replace("Wikipedia: ", "") if lines else "article"
        return f"Wikipedia: {title}"
    return result[:80]


def _stats_md(data: dict) -> str:
    if not data:
        return "*Run a query to see stats.*"
    iters   = data.get("iterations", 0)
    n_tools = len(data.get("steps", []))
    secs    = data.get("time_ms", 0) / 1000
    tools_used = list(dict.fromkeys(s["tool"] for s in data.get("steps", [])))
    badges = "  ".join(
        f"{TOOL_META[t]['emoji']} {TOOL_META[t]['label']}" for t in tools_used
    ) or "—"
    return f"**{iters}** steps &nbsp;·&nbsp; **{n_tools}** tool calls &nbsp;·&nbsp; **{secs:.1f}s**\n\n{badges}"


# ── Streaming chat function ─────────────────────────────────────────────────

def respond(message: str, display_history: list, api_history: list, api_key_input: str):
    """
    Generator — streams agent steps to the Gradio Chatbot.

    Two separate histories:
      display_history — what the chatbot shows (includes tool call blocks, emojis)
      api_history     — clean Q&A pairs sent to the LLM for multi-turn context
    """
    if not message.strip():
        yield display_history, display_history, api_history, _stats_md({}), ""
        return

    api_key = ENV_KEY or api_key_input.strip()
    if not api_key:
        err = (
            "❌ **No API key configured.**\n\n"
            "Enter your **GROQ API Key** in the sidebar (free at [console.groq.com](https://console.groq.com))."
        )
        new_disp = display_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": err},
        ]
        yield new_disp, new_disp, api_history, _stats_md({}), ""
        return

    # Append user message + loading placeholder to display history
    working = display_history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ *Analyzing your question...*"},
    ]
    yield working, display_history, api_history, "⏳ Running...", ""

    tool_lines: list[str] = []

    try:
        for step in agent_stream(message, api_key, conversation_history=api_history):

            if step["type"] == "tool_start":
                tool    = step["tool"]
                emoji   = TOOL_META[tool]["emoji"]
                label   = TOOL_META[tool]["label"]
                preview = step["arg_preview"]
                line = f"> {emoji} **{label}** {preview}  \n> ⏳ *Fetching...*"
                tool_lines.append(line)
                partial = "🤔 *Reasoning with tools...*\n\n" + "\n\n".join(tool_lines)
                working[-1] = {"role": "assistant", "content": partial}
                yield working, display_history, api_history, "⏳ Running...", ""

            elif step["type"] == "tool_result":
                tool    = step["tool"]
                result  = step["result"]
                preview = _result_preview(tool, result)
                if tool_lines:
                    base = tool_lines[-1].split("\n")[0]
                    tool_lines[-1] = base + f"  \n> ✅ *{preview}*"
                partial = "🤔 *Reasoning with tools...*\n\n" + "\n\n".join(tool_lines)
                working[-1] = {"role": "assistant", "content": partial}
                yield working, display_history, api_history, "⏳ Running...", ""

            elif step["type"] == "final":
                answer = step["answer"]
                if tool_lines:
                    done_section  = "\n\n".join(tool_lines)
                    final_content = f"🤔 *Reasoning with tools...*\n\n{done_section}\n\n---\n\n{answer}"
                else:
                    final_content = answer

                working[-1] = {"role": "assistant", "content": final_content}
                saved_display = display_history + [
                    {"role": "user",      "content": message},
                    {"role": "assistant", "content": final_content},
                ]
                # Store only the clean answer (no tool blocks) for API context
                saved_api = api_history + [
                    {"role": "user",      "content": message},
                    {"role": "assistant", "content": answer},
                ]
                yield working, saved_display, saved_api, _stats_md(step), ""

    except Exception as e:
        err_msg = f"❌ **Error:** {e}\n\nCheck your API key or try again."
        working[-1] = {"role": "assistant", "content": err_msg}
        yield working, display_history, api_history, _stats_md({}), ""


def clear_chat():
    return [], [], [], _stats_md({}), ""


# ── Build UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Research Agent") as demo:

        # Header
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── Left: Chat ───────────────────────────────────────────────
            with gr.Column(scale=3, min_width=480):
                chatbot = gr.Chatbot(
                    value=[],
                    label="",
                    height=520,
                    layout="bubble",
                    render_markdown=True,
                    placeholder=PLACEHOLDER,
                    buttons=["copy"],
                    sanitize_html=False,  # allow our <br> in blockquotes
                )

                with gr.Row(elem_classes=["input-area"]):
                    msg_box = gr.Textbox(
                        placeholder="Ask anything — I'll search the web, calculate, and explain...",
                        show_label=False,
                        scale=6,
                        lines=1,
                        max_lines=5,
                        autofocus=True,
                        submit_btn=False,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=90)

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear", size="sm", variant="secondary")
                    gr.Markdown(
                        "*Enter to send · Shift+Enter for newline*",
                        elem_classes=["gr-text-sm"],
                    )

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_box,
                    label="💡 Click an example to try",
                    examples_per_page=4,
                )

            # ── Right: Sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):

                # API Key (hidden if set via environment secret)
                if not ENV_KEY:
                    with gr.Group(elem_classes=["key-panel"]):
                        gr.Markdown("### 🔑 API Key")
                        api_key_box = gr.Textbox(
                            label="GROQ API Key",
                            placeholder="gsk_...",
                            type="password",
                            info="Free at console.groq.com",
                        )
                else:
                    gr.Markdown("> ✅ **API key configured** — ready to go!")
                    api_key_box = gr.Textbox(value="", visible=False)

                # Tool cards
                gr.Markdown("### 🛠 Available Tools")
                for emoji, label, desc in TOOL_CARDS:
                    gr.HTML(f"""
                    <div class="tool-card">
                        <strong>{emoji} {label}</strong><br>
                        <span style="color:#6b7280;font-size:0.82rem">{desc}</span>
                    </div>""")

                # Stats
                gr.Markdown("### 📊 Last Run")
                stats_md = gr.Markdown(
                    "*Run a query to see stats.*",
                    elem_classes=["stats-panel"],
                )

                # How it works (collapsible)
                with gr.Accordion("⚙️ How it works", open=False):
                    gr.Markdown("""
**ReAct loop:**
1. LLM reads your question
2. Decides which tool to call
3. Observes the result
4. Repeats until ready to answer

**Why better than plain LLM:**
- Searches *live* web (not stale training data)
- Does *exact* math (no hallucinated numbers)
- Cites sources for key facts

**Model:** Qwen3-32B via Groq (~800 tok/s)
""")

        # ── State ────────────────────────────────────────────────────────
        # display_state: full chat history with tool-call blocks (for chatbot)
        # api_state:     clean Q&A pairs only (for LLM multi-turn context)
        display_state = gr.State([])
        api_state     = gr.State([])

        # ── Events ───────────────────────────────────────────────────────
        inputs  = [msg_box, display_state, api_state, api_key_box]
        outputs = [chatbot, display_state, api_state, stats_md, msg_box]

        msg_box.submit(respond, inputs, outputs)
        send_btn.click(respond, inputs, outputs)
        clear_btn.click(clear_chat, None, outputs)

    return demo


# ── Entry point ─────────────────────────────────────────────────────────────

demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CSS,
        share=False,
    )
