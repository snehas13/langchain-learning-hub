"""
USE CASE 5 — Antigravity Showcase (Streaming + Structured Output)
=================================================================
CONCEPTS:
  Streaming        : Print tokens as they arrive (not wait for full response)
  JsonOutputParser : Force LLM to return valid JSON
  Pydantic schema  : Define the exact shape of the JSON you want

antigravity FULL CIRCLE:
  This use case IS about antigravity. We:
  1. Import it (launches the XKCD comic)
  2. Use RAG knowledge about it
  3. Stream an explanation token by token
  4. Extract structured data about Python Easter eggs as JSON
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import antigravity  # 🚀 launches https://xkcd.com/353/

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils.config import get_llm, banner, info, success


def run():
    banner("Use Case 5 — Streaming + Structured Output (Antigravity!)", "cyan")

    llm = get_llm(temperature=0.7)

    # ── Part A: STREAMING ─────────────────────────────────────────────────────
    # Instead of waiting for the full response, .stream() yields tokens one by one
    # Great for chatbots — users see output immediately, feels responsive
    info("Part A → Streaming response (tokens appear as they generate)\n")

    stream_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enthusiastic Python historian."),
        ("human",  "{question}"),
    ])

    stream_chain = stream_prompt | llm | StrOutputParser()

    print("  🌊 Streaming: ", end="", flush=True)
    for token in stream_chain.stream({
        "question": (
            "Tell me the full story of 'import antigravity' — "
            "who created it, when, why, and what it teaches us about Python culture. "
            "Keep it to 3 sentences."
        )
    }):
        print(token, end="", flush=True)  # print each token immediately
    print("\n")

    # ── Part B: STRUCTURED OUTPUT (JSON) ──────────────────────────────────────
    # Sometimes you need the LLM to return structured data, not prose.
    # JsonOutputParser tells the LLM to respond ONLY with valid JSON.
    info("Part B → Structured JSON output\n")

    json_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Python documentation bot. "
            "Always respond with valid JSON only. No prose, no explanation. "
            "Return exactly this structure:\n"
            "{{\n"
            '  "easter_eggs": [\n'
            '    {{"name": "...", "command": "...", "description": "...", "year_added": ...}}\n'
            "  ]\n"
            "}}"
        )),
        ("human", "List {count} Python Easter eggs as JSON."),
    ])

    json_chain = json_prompt | llm | JsonOutputParser()

    result = json_chain.invoke({"count": 3})

    print("  📦 Structured JSON response:")
    print(json.dumps(result, indent=4))
    print()

    # ── Part C: BATCH CALLS ──────────────────────────────────────────────────
    # .batch() runs multiple inputs in parallel — much faster than sequential invoke()
    info("Part C → Batch processing (multiple inputs at once)\n")

    batch_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer in exactly one sentence."),
        ("human",  "{question}"),
    ])
    batch_chain = batch_prompt | llm | StrOutputParser()

    questions = [
        {"question": "What does 'import this' do in Python?"},
        {"question": "What year was Python created?"},
        {"question": "What is PEP 8?"},
    ]

    answers = batch_chain.batch(questions)

    print("  ⚡ Batch results (all processed in parallel):")
    for q, a in zip(questions, answers):
        print(f"    Q: {q['question']}")
        print(f"    A: {a}\n")

    success("Lesson: stream() for UX, JsonOutputParser for structured data, batch() for speed!")


if __name__ == "__main__":
    run()