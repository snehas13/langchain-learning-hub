"""
config.py — Central configuration for LangChain Learning Hub
Every use case imports get_llm() from here so API setup is in one place.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage


# --------------------
# Load environment variables
# --------------------
load_dotenv()


# --------------------
# Groq model accessor
# --------------------
def get_llm(temperature: float = 0.7):
    """
    Returns a ready-to-use Groq LLM instance.

    temperature:
      0.0 = deterministic, factual
      0.7 = balanced (default)
      1.0 = creative, varied
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "\n❌  GROQ_API_KEY not set!\n"
            "    1. Copy .env.example to .env\n"
            "    2. Get a free key from https://console.groq.com\n"
        )


    return ChatGroq(model= "llama-3.1-8b-instant", temperature = temperature, api_key=api_key)

COLORS = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "purple": "\033[95m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
}

# --------------------
# Console UI helpers
# --------------------
def banner(title: str, color: str = "cyan") -> None:
    c = COLORS.get(color, "")
    r = COLORS["reset"]
    b = COLORS["bold"]
    print(f"\n{c}{b}{'═' * 60}{r}")
    print(f"{c}{b}  {title}{r}")
    print(f"{c}{b}{'═' * 60}{r}\n")


def info(msg: str) -> None:
    print(f"  {COLORS['yellow']}ℹ  {msg}{COLORS['reset']}")


def success(msg: str) -> None:
    print(f"  {COLORS['green']}✓  {msg}{COLORS['reset']}")