"""
main.py — LangChain Learning Hub Entry Point
=============================================
Run individual use cases or all of them.
Each use case teaches a core LangChain concept.

Usage:
  python main.py          → shows the menu
  python main.py all      → runs all use cases in order
  python main.py 1        → runs use case 1 only
"""

import sys
import os

# Add project root to path so 'utils' imports work
sys.path.insert(0, os.path.dirname(__file__))


def show_menu():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🚀 LangChain Learning Hub — with Groq           ║
╠══════════════════════════════════════════════════════════╣
║  1  → Simple Chain (Prompt | LLM | Parser)               ║
║  2  → Conversation with Memory (ChatBot)                  ║
║  3  → RAG Pipeline (Search your own docs)                 ║
║  4  → Tools and Agents (LLM as decision-maker)           ║
║  5  → Antigravity Showcase (Streaming + JSON output)      ║
║  all → Run all use cases in order                         ║
╚══════════════════════════════════════════════════════════╝
""")


def run_use_case(num: str):
    import importlib
    mapping = {
        "1": "use_cases.01_simple_chain",
        "2": "use_cases.02_conversation_memory",
        "3": "use_cases.03_rag_pipeline",
        "4": "use_cases.04_tools_agent",
        "5": "use_cases.05_antigravity_showcase",
    }
    if num not in mapping:
        print(f"  ❌ Unknown use case: {num}. Choose 1-5 or 'all'")
        return
    module = importlib.import_module(mapping[num])
    module.run()

def main():
    if len(sys.argv) < 2:
        show_menu()
        choice = input("  Enter choice (1-5 or all): ").strip()
    else:
        choice = sys.argv[1]

    if choice == "all":
        for i in range(1, 6):
            run_use_case(str(i))
            print("\n" + "─" * 60 + "\n")
    else:
        run_use_case(choice)


if __name__ == "__main__":
    main()