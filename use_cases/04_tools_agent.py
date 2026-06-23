"""
USE CASE 4 — Tools and Agents
==============================
CONCEPTS:
  @tool decorator  : Turns a Python function into a LangChain tool
  create_react_agent: Creates a ReAct-style agent (Reason + Act loop)
  AgentExecutor    : Runs the agent loop until it reaches a final answer

HOW AGENTS WORK (ReAct loop):
  1. LLM reads the question + list of available tools
  2. LLM decides: "I should use tool X with input Y"
  3. Tool runs, result returned to LLM
  4. LLM decides: "Now I have enough info" OR "I need another tool"
  5. Loop repeats until LLM produces a Final Answer

WHY AGENTS VS CHAINS?
  Chains = fixed sequence (always prompt → llm → parse)
  Agents = dynamic sequence (LLM decides what to do at each step)

antigravity CONNECTION:
  One of our tools explains Python Easter eggs — the LLM decides
  when it needs that tool based on the question asked.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from utils.config import get_llm, banner, info, success


# ── Define Tools ─────────────────────────────────────────────────────────────
# @tool turns a Python function into a LangChain tool.
# The docstring becomes the tool description (the LLM reads it to decide when to use it).
# The function signature becomes the tool's input schema.

@tool
def get_python_easter_eggs(topic: str) -> str:
    """
    Returns information about Python Easter eggs and hidden features.
    Use this when the user asks about fun Python secrets, Easter eggs,
    'import antigravity', 'import this', or hidden Python tricks.
    Input: a topic like 'antigravity' or 'zen of python'
    """
    easter_eggs = {
        "antigravity": (
            "import antigravity — Added by Barry Warsaw in 2008. "
            "It opens https://xkcd.com/353/ which shows Python can do anything, "
            "even make you fly! It embodies Python's fun, approachable spirit."
        ),
        "this": (
            "import this — Prints the Zen of Python (PEP 20) by Tim Peters. "
            "19 aphorisms like 'Beautiful is better than ugly', "
            "'Readability counts', 'Simple is better than complex'. "
            "The source code of the module is itself an Easter egg — it's a ROT-13 cipher!"
        ),
        "hello": (
            "__hello__ module — import __hello__ prints 'Hello World!'. "
            "A nod to the classic first program every developer writes."
        ),
    }
    topic_lower = topic.lower()
    for key, val in easter_eggs.items():
        if key in topic_lower:
            return val
    return f"I have info on: antigravity, this (Zen of Python), hello. You asked about: {topic}"


@tool
def calculate_python_stats(calculation: str) -> str:
    """
    Performs simple calculations related to Python statistics.
    Use this when the user asks to compute or calculate something numerical.
    Input: a simple arithmetic expression like '2 + 2' or '10 * 5'
    """
    try:
        # safe eval for arithmetic only
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in calculation):
            return "Only arithmetic operations allowed for safety."
        result = eval(calculation, {"__builtins__": {}})
        return f"Result of '{calculation}' = {result}"
    except Exception as e:
        return f"Could not calculate: {e}"


@tool
def get_python_version_info(version: str) -> str:
    """
    Returns key features and release info for a Python version.
    Use this when the user asks about Python version history or features.
    Input: a version string like '3.10' or '3.12'
    """
    versions = {
        "3.10": "Python 3.10 (2021): Structural pattern matching (match/case), better error messages",
        "3.11": "Python 3.11 (2022): 60% faster than 3.10, fine-grained error locations in tracebacks",
        "3.12": "Python 3.12 (2023): Even faster, better f-strings, 'import antigravity' still works!",
        "2.7":  "Python 2.7 (EOL 2020): The last Python 2 release. Upgrade to 3!",
    }
    return versions.get(version, f"No info for Python {version}. Try: 3.10, 3.11, 3.12")


def run():
    banner("Use Case 4 — Tools and Agents", "yellow")

    tools = [get_python_easter_eggs, calculate_python_stats, get_python_version_info]
    llm = get_llm(temperature=0)  # temperature=0 for reliable tool selection

    # STEP 1 — The agent needs a specific prompt format
    # {tools}         → list of tool names + descriptions
    # {tool_names}    → comma-separated tool names
    # {input}         → the user's question
    # {agent_scratchpad} → the agent's internal reasoning + tool results
    info("Step 1 → Defining ReAct agent prompt")
    react_prompt = PromptTemplate.from_template("""
Answer the following question using the available tools.

You have access to these tools:
{tools}

Use this EXACT format:
Question: the input question
Thought: think about what to do
Action: the tool name (one of [{tool_names}])
Action Input: the input to the tool
Observation: the tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    # STEP 2 — Create agent
    # create_react_agent gives the LLM the tools list and prompt
    # The LLM then reasons about which tool to use at each step
    info("Step 2 → Creating ReAct agent")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    # STEP 3 — AgentExecutor runs the loop
    # max_iterations=5 prevents infinite loops
    # handle_parsing_errors=True prevents crashes on bad LLM output
    info("Step 3 → Wrapping in AgentExecutor")
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,       # prints each Thought/Action/Observation step!
        max_iterations=5,
        handle_parsing_errors=True,
    )

    # STEP 4 — Ask questions (agent decides which tools to use)
    info("Step 4 → Asking questions — watch the agent's reasoning!\n")
    questions = [
        "What happens when you run import antigravity in Python?",
        "What are the key features of Python 3.12 and what is 15 * 8?",
    ]

    for q in questions:
        print(f"\n  ❓ Question: {q}")
        print("  " + "─" * 55)
        result = executor.invoke({"input": q})
        print(f"\n  ✅ Final Answer: {result['output']}\n")

    success("Lesson: Agents = LLM as a decision-maker, not just a text generator!")


if __name__ == "__main__":
    run()