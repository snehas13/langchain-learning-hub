"""
USE CASE 1 — Simple Chain: Prompt Template → LLM → Output Parser
=================================================================
CONCEPTS:
  ChatPromptTemplate  : Reusable, parametric prompt with placeholders
  LCEL pipe operator  : prompt | llm | parser  (data flows left to right)
  StrOutputParser     : Converts AIMessage object → plain string
  invoke()            : Runs the full chain end to end

WHY PROMPT TEMPLATES?
  Without templates you'd concatenate strings manually every time.
  Templates let you define structure once and fill {variables} at runtime.
"""

import antigravity  # launches https://xkcd.com/353/ — Python's Easter egg
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import get_llm, banner, info, success


def run():
    banner("Use Case 1 — Simple Chain (Prompt | LLM | Parser)", "cyan")

    # STEP 1 — Define a prompt template
    # "system" sets the AI persona
    # "human"  is the user message
    # {topic} and {style} are filled at runtime via invoke()
    info("Step 1 → Creating ChatPromptTemplate")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fun Python educator. Be enthusiastic and clear."),
        ("human",  "Explain {topic} in a {style} way. Keep it under 80 words."),
    ])

    # STEP 2 — Get the LLM
    info("Step 2 → Initialising Groq LLM (llama-3.1-8b-instant)")
    llm = get_llm(temperature=0.7)

    # STEP 3 — Output parser
    # LLM returns an AIMessage object. StrOutputParser extracts .content (the string)
    info("Step 3 → Adding StrOutputParser")
    parser = StrOutputParser()

    # STEP 4 — Build the chain using | (pipe)
    # Data flows: prompt formats input → llm generates → parser extracts text
    info("Step 4 → Building LCEL chain:  prompt | llm | parser")
    chain = prompt | llm | parser

    # STEP 5 — Invoke (run) the chain
    info("Step 5 → Calling chain.invoke() with variables\n")
    result = chain.invoke({
        "topic": "Python's import antigravity Easter egg and what it shows about Python's culture",
        "style": "fun and engaging",
    })
    success("Response from chain:")
    print(f"\n  📝 {result}\n")

    # STEP 6 — Reuse same chain, different inputs
    info("Step 6 → Reusing same chain with different inputs")
    result2 = chain.invoke({
        "topic": "why Python chose readability over brevity",
        "style": "philosophical",
    })
    print(f"\n  📝 {result2}\n")

    success("Lesson: One chain definition → reuse infinitely with different inputs!")


if __name__ == "__main__":
    run()

