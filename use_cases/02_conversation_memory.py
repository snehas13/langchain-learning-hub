"""
USE CASE 2 — Conversation with Memory
======================================
CONCEPTS:
  ConversationBufferMemory : Stores the full conversation history as a list
  ConversationChain        : Chain that auto-loads + saves memory each turn
  MessagesPlaceholder      : Slot in the prompt where history is injected

THE PROBLEM WITHOUT MEMORY:
  Q: "What is Python?"   → AI answers
  Q: "Who invented it?"  → AI has no idea what "it" is (stateless!)

WITH MEMORY:
  The full transcript is passed to the LLM each time, so "it" has context.

MEMORY TYPES IN LANGCHAIN:
  ConversationBufferMemory        → stores everything (simple)
  ConversationSummaryMemory       → summarises old messages to save tokens
  ConversationBufferWindowMemory  → keeps last N messages only
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.config import get_llm, banner, info, success


def run():
    banner("Use Case 2 — Chatbot with Memory", "purple")

    # STEP 1 — Create memory
    # memory_key="history" → the prompt template uses {history} to inject messages
    # return_messages=True → stores Message objects (not plain string)
    info("Step 1 → Creating ConversationBufferMemory")
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )

    # STEP 2 — Prompt with MessagesPlaceholder
    # MessagesPlaceholder(variable_name="history") is replaced by the full
    # conversation history stored in memory at runtime
    info("Step 2 → Building prompt with MessagesPlaceholder")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are PythonBot, an expert Python tutor who loves Easter eggs. "
            "You know 'import antigravity' opens an XKCD comic showing Python "
            "can do anything — even fly! Remember everything the student says."
        )),
        MessagesPlaceholder(variable_name="history"),  # ← memory injected here
        ("human", "{input}"),
    ])

    # STEP 3 — ConversationChain ties it all together
    # Each invoke() call:
    #   1. Loads history from memory into prompt
    #   2. Sends prompt + new message to LLM
    #   3. Saves new exchange back to memory automatically
    info("Step 3 → Creating ConversationChain (LLM + Prompt + Memory)")
    chain = ConversationChain(
        llm=get_llm(temperature=0.6),
        memory=memory,
        prompt=prompt,
        verbose=False,  # set True to print the full prompt each turn
    )

    # STEP 4 — Multi-turn conversation (notice how context carries forward)
    info("Step 4 → Running a 4-turn conversation\n")
    turns = [
        "Hi! I just ran 'import antigravity' in Python. What happened?",
        "That's cool! Who created that Easter egg?",          # 'that' needs context
        "Is there another Easter egg about Python's philosophy?",
        "Based on all this, what does Python value most?",    # needs full context
    ]

    for i, msg in enumerate(turns, 1):
        print(f"  👤 Turn {i} → {msg}")
        response = chain.invoke({"input": msg})
        print(f"  🤖 PythonBot → {response['response']}\n")

    # STEP 5 — Inspect memory
    info("Step 5 → What is stored in memory?")
    msgs = memory.chat_memory.messages
    print(f"\n  Total messages stored: {len(msgs)}")
    for m in msgs[-4:]:
        role = "Human" if "Human" in type(m).__name__ else "AI"
        print(f"    [{role}]: {m.content[:80]}...")

    success("Lesson: Memory = full transcript injected into every LLM call!")


if __name__ == "__main__":
    run()