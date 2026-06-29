import math
import datetime
import requests
from ddgs import DDGS

# ── Tool implementations ────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: {query}"
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['title']}\n{r['body']}\nSource: {r['href']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


def calculate(expression: str) -> str:
    try:
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_env.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def get_current_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y — %H:%M:%S UTC+0")


def wikipedia_search(topic: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "AI-Research-Agent/2.0"})
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            title = data.get("title", topic)
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            return f"Wikipedia: {title}\n{page_url}\n\n{extract[:2000]}"
        return f"Wikipedia has no article for: '{topic}'. Try a more general term."
    except Exception as e:
        return f"Wikipedia error: {e}"


# ── Tool schemas (sent to LLM so it knows when/how to use each tool) ────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current information: news, prices, salaries, "
                "company info, recent events, job requirements. Always prefer this when "
                "accuracy matters or information might have changed recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific, well-formed search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 8)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression accurately. Supports arithmetic, "
                "percentages, exponents (**), sqrt, log, sin, cos, pi, e, etc. "
                "Always use this for any numeric computation — never guess math in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Valid Python math expression, e.g. '5600 * 12' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time. Use when the user asks about today, current year, or time-sensitive comparisons.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Look up a concept, person, place, technology, or scientific topic on Wikipedia. "
                "Best for definitions, historical background, technical explanations. "
                "Not good for current prices or very recent events — use web_search for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The Wikipedia page title or topic to look up"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "web_search": web_search,
    "calculate": calculate,
    "get_current_date": get_current_date,
    "wikipedia_search": wikipedia_search,
}

TOOL_META = {
    "web_search":      {"emoji": "🔍", "label": "Web Search"},
    "calculate":       {"emoji": "🧮", "label": "Calculator"},
    "get_current_date":{"emoji": "📅", "label": "Date & Time"},
    "wikipedia_search":{"emoji": "📖", "label": "Wikipedia"},
}


def execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"
    return TOOL_REGISTRY[name](**args)
