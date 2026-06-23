# 🚀 LangChain Learning Hub

A hands-on learning project built with **LangChain** and **Groq (free LLM API)**
that teaches every major LangChain concept through 5 progressive use cases —
all tied together with Python's famous `import antigravity` Easter egg.

---

## 📋 Table of Contents

- [What is LangChain?](#what-is-langchain)
- [Why LangChain?](#why-langchain)
- [LangChain Architecture](#langchain-architecture)
- [Core Concepts](#core-concepts)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Use Cases](#use-cases)
  - [Use Case 1 — Simple Chain](#use-case-1--simple-chain)
  - [Use Case 2 — Conversation with Memory](#use-case-2--conversation-with-memory)
  - [Use Case 3 — RAG Pipeline](#use-case-3--rag-pipeline)
  - [Use Case 4 — Tools and Agents](#use-case-4--tools-and-agents)
  - [Use Case 5 — Streaming and Structured Output](#use-case-5--streaming-and-structured-output)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Key Learnings](#key-learnings)

---

## What is LangChain?

**LangChain** is an open-source framework for building applications powered by
Large Language Models (LLMs).

A raw LLM (like GPT-4, Llama 3, or Gemini) is essentially a function —
it takes text in and returns text out. That is powerful on its own, but it
cannot remember previous conversations, search your documents, call APIs,
or decide what action to take next. LangChain solves all of that.

```
Raw LLM   =  an engine sitting on a workbench
LangChain =  the full car — steering, GPS, brakes, and fuel system included
```

LangChain lets you connect LLMs to:

- **Your own data** — PDFs, text files, databases, websites
- **Memory** — so the LLM remembers what you said earlier
- **Tools** — calculators, search engines, APIs, custom Python functions
- **Other LLMs** — chain multiple models together
- **Structured output** — force the LLM to return valid JSON

---

## Why LangChain?

### Without LangChain

Every time you build an LLM feature you manually:
- Construct prompt strings with f-strings
- Parse the response text yourself
- Manage conversation history in a list and inject it into every request
- Write retry logic for API failures
- Build your own document search system from scratch
- Handle tool calling, output validation, and error recovery yourself

### With LangChain

```python
# A complete conversational RAG pipeline in a few lines
chain = prompt | llm | parser
result = chain.invoke({"topic": "Python", "style": "fun"})
```

| Problem | LangChain Solution |
|---|---|
| Prompt management | `ChatPromptTemplate` with typed variables |
| Model swapping | One-line change, same interface everywhere |
| Conversation history | `ConversationBufferMemory` — automatic |
| Document search | `FAISS` + `TextLoader` + `TextSplitter` |
| Tool calling | `@tool` decorator + `AgentExecutor` |
| Streaming | `.stream()` method on any chain |
| Structured output | `JsonOutputParser` |
| Composability | LCEL `\|` pipe operator |

---

## LangChain Architecture

### Overall Framework Architecture

```mermaid
graph TD
    User([👤 User Input]) --> PT[Prompt Template\nFills variables into messages]
    PT --> LLM[LLM / Chat Model\nGroq · OpenAI · Gemini · Ollama]
    LLM --> OP[Output Parser\nString · JSON · Pydantic]
    OP --> OUT([✅ Final Output])

    MEM[(Memory\nConversation History)] -- load history --> PT
    LLM -- save exchange --> MEM

    VS[(Vector Store\nFAISS · Chroma · Pinecone)] -- relevant chunks --> PT
    DOCS[📄 Documents] -- embed + index --> VS

    TOOLS[🔧 Tools\nAPIs · Functions · Search] -- tool result --> LLM
    LLM -- tool call --> TOOLS

    style User fill:#4A4A4A,color:#fff
    style OUT fill:#2D6A4F,color:#fff
    style LLM fill:#1B4F72,color:#fff
    style PT fill:#6C3483,color:#fff
    style OP fill:#1A5276,color:#fff
    style MEM fill:#7D6608,color:#fff
    style VS fill:#922B21,color:#fff
    style TOOLS fill:#784212,color:#fff
    style DOCS fill:#4A4A4A,color:#fff
```

### LCEL — How the Pipe Operator Works

```mermaid
graph LR
    A["invoke({'topic':'Python',\n'style':'fun'})"] -->|dict| B[ChatPromptTemplate\nFormats into messages]
    B -->|ChatPromptValue| C[ChatGroq LLM\nCalls Groq API]
    C -->|AIMessage object| D[StrOutputParser\nExtracts .content]
    D -->|plain string| E["'Python is great\nbecause...'"]

    style A fill:#4A4A4A,color:#fff
    style B fill:#6C3483,color:#fff
    style C fill:#1B4F72,color:#fff
    style D fill:#1A5276,color:#fff
    style E fill:#2D6A4F,color:#fff
```

---

## Core Concepts

### 1. Prompt Template

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",  "Explain {topic} in a {style} way."),
])
```

```mermaid
graph LR
    V1["{topic} = 'antigravity'"] --> PT[ChatPromptTemplate]
    V2["{style} = 'fun'"] --> PT
    PT --> MSG["SystemMessage: You are a helpful assistant\nHumanMessage: Explain antigravity in a fun way"]

    style PT fill:#6C3483,color:#fff
    style MSG fill:#1B4F72,color:#fff
```

Without templates you concatenate strings manually everywhere.
Templates make prompts testable, versionable, and reusable across your entire app.

### 2. LLM / Chat Model

```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
```

`temperature` controls randomness:
- `0.0` = deterministic, factual answers
- `0.7` = balanced — default for most tasks
- `1.0` = creative, varied output

LangChain supports every major provider through the same interface.
Swap `ChatGroq` for `ChatOpenAI` or `ChatAnthropic` — nothing else changes.

### 3. Memory Types

```mermaid
graph TD
    M[Memory Options] --> BUF[ConversationBufferMemory\nStores ALL messages\nSimple, no limits]
    M --> SUM[ConversationSummaryMemory\nSummarises old messages\nSaves tokens on long chats]
    M --> WIN[ConversationBufferWindowMemory\nKeeps last N messages only\nSliding window]
    M --> TOK[ConversationTokenBufferMemory\nTrims by token count\nFits within context window]

    style M fill:#7D6608,color:#fff
    style BUF fill:#4A4A4A,color:#fff
    style SUM fill:#4A4A4A,color:#fff
    style WIN fill:#4A4A4A,color:#fff
    style TOK fill:#4A4A4A,color:#fff
```

### 4. RAG — Retrieval Augmented Generation

```mermaid
graph TD
    DOC[📄 Your Document\ndocs/python_knowledge.txt] --> LOAD[TextLoader\nWraps in Document objects]
    LOAD --> SPLIT[RecursiveCharacterTextSplitter\nBreaks into 500-char chunks\nwith 50-char overlap]
    SPLIT --> EMBED[HuggingFaceEmbeddings\nConverts each chunk to\na 384-dimension vector]
    EMBED --> STORE[(FAISS Vector Store\nStores all chunk vectors)]

    Q([❓ User Question]) --> QEMBED[Embed the question\ninto a vector]
    QEMBED --> SEARCH[Similarity Search\nFind top-3 nearest chunks]
    STORE --> SEARCH
    SEARCH --> CONTEXT[Retrieved chunks\ninjected as context]
    CONTEXT --> PROMPT[RAG Prompt\nsystem + context + question]
    Q --> PROMPT
    PROMPT --> LLM[Groq LLM\nAnswers using context]
    LLM --> ANS([✅ Accurate Answer\nbased on your document])

    style DOC fill:#4A4A4A,color:#fff
    style STORE fill:#922B21,color:#fff
    style Q fill:#2D6A4F,color:#fff
    style ANS fill:#2D6A4F,color:#fff
    style LLM fill:#1B4F72,color:#fff
    style PROMPT fill:#6C3483,color:#fff
```

### 5. Agents — ReAct Loop

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant E as AgentExecutor
    participant L as LLM (Groq)
    participant T as 🔧 Tools

    U->>E: "What is antigravity and what is 15 * 8?"
    E->>L: Question + list of available tools
    L->>E: Thought: I need easter egg info first
    L->>E: Action: get_python_easter_eggs("antigravity")
    E->>T: Call get_python_easter_eggs("antigravity")
    T->>E: "Opens xkcd.com/353, added by Barry Warsaw..."
    E->>L: Observation: tool result
    L->>E: Thought: Now I need to calculate
    L->>E: Action: calculate_python_stats("15 * 8")
    E->>T: Call calculate_python_stats("15 * 8")
    T->>E: "Result of '15 * 8' = 120"
    E->>L: Observation: 120
    L->>E: Thought: I now have both answers
    L->>E: Final Answer: antigravity opens XKCD... and 120
    E->>U: ✅ Final Answer
```

---

## Project Structure

```
langchain_learning_hub/
│
├── .env                           # Your API key (never commit this)
├── .env.example                   # Template for others to copy
├── .gitignore                     # Excludes venv/, .env, __pycache__
├── requirements.txt               # All dependencies with versions
├── main.py                        # Entry point and menu dispatcher
│
├── utils/
│   ├── __init__.py                # Exports get_llm, banner, info, success
│   └── config.py                  # Groq LLM setup, CLI colour helpers
│
├── use_cases/
│   ├── __init__.py                # Marks folder as Python package
│   ├── 01_simple_chain.py         # Prompt | LLM | Parser (LCEL basics)
│   ├── 02_conversation_memory.py  # Memory + ConversationChain
│   ├── 03_rag_pipeline.py         # Document loading, FAISS, RAG chain
│   ├── 04_tools_agent.py          # @tool, ReAct agent, AgentExecutor
│   └── 05_antigravity_showcase.py # Streaming, JSON output, batch()
│
└── docs/
    └── python_knowledge.txt       # Knowledge base used in RAG use case
```

### Module Dependency Flow

```mermaid
graph TD
    MAIN[main.py\nEntry point + menu] --> UC1[01_simple_chain.py]
    MAIN --> UC2[02_conversation_memory.py]
    MAIN --> UC3[03_rag_pipeline.py]
    MAIN --> UC4[04_tools_agent.py]
    MAIN --> UC5[05_antigravity_showcase.py]

    UC1 --> CFG[utils/config.py\nget_llm · banner · info · success]
    UC2 --> CFG
    UC3 --> CFG
    UC4 --> CFG
    UC5 --> CFG

    CFG --> GROQ[ChatGroq\nGroq API]
    CFG --> ENV[.env\nGROQ_API_KEY]

    UC3 --> DOCS[docs/python_knowledge.txt\nKnowledge base]

    style MAIN fill:#4A4A4A,color:#fff
    style CFG fill:#6C3483,color:#fff
    style GROQ fill:#1B4F72,color:#fff
    style ENV fill:#922B21,color:#fff
    style DOCS fill:#2D6A4F,color:#fff
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- A free Groq API key from [https://console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/snehas13/langchain-learning-hub.git
cd langchain-learning-hub
```

### Step 2 — Create and activate a virtual environment

A virtual environment isolates this project's dependencies from your
global Python installation. Always use one per project.

```bash
# Create the virtual environment
python -m venv venv
```

```bash
# Activate — Mac / Linux
source venv/bin/activate

# Activate — Windows Command Prompt
venv\Scripts\activate.bat

# Activate — Windows PowerShell
venv\Scripts\Activate.ps1
```

You will see `(venv)` at the start of your terminal prompt. That means it is active.

```mermaid
graph LR
    A[python -m venv venv\nCreate isolated environment] --> B[source venv/bin/activate\nActivate it]
    B --> C[pip install -r requirements.txt\nInstall only into venv]
    C --> D[python main.py\nRun the app]
    D --> E[deactivate\nWhen done]

    style A fill:#4A4A4A,color:#fff
    style B fill:#2D6A4F,color:#fff
    style C fill:#1B4F72,color:#fff
    style D fill:#6C3483,color:#fff
    style E fill:#922B21,color:#fff
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Use Case 3 (RAG) downloads a ~90MB HuggingFace embeddings model
> (`all-MiniLM-L6-v2`) on first run. This is free and runs locally —
> no API key needed for embeddings.

### Step 4 — Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at [https://console.groq.com](https://console.groq.com) — no credit card required.

---

## Use Cases

### Use Case 1 — Simple Chain

**File:** `use_cases/01_simple_chain.py`
**Concepts:** `ChatPromptTemplate` · LCEL `|` pipe · `StrOutputParser` · `invoke()`

#### What it does

Builds the most fundamental LangChain pattern — a prompt template connected
to an LLM connected to an output parser using the `|` pipe operator.
Demonstrates `import antigravity` (Python's Easter egg that opens
[xkcd.com/353](https://xkcd.com/353/)) and asks the LLM to explain
its cultural significance.

#### Data Flow

```mermaid
graph LR
    IN["invoke({\n  topic: 'antigravity',\n  style: 'fun'\n})"] --> PT

    PT["ChatPromptTemplate\nFills variables into\nsystem + human messages"] --> LLM

    LLM["ChatGroq\nSends to Groq API\nReturns AIMessage object"] --> OP

    OP["StrOutputParser\nExtracts .content\nfrom AIMessage"] --> OUT

    OUT["'Python's antigravity\nmodule opens a comic\nshowing Python can fly!'"]

    style IN fill:#4A4A4A,color:#fff
    style PT fill:#6C3483,color:#fff
    style LLM fill:#1B4F72,color:#fff
    style OP fill:#1A5276,color:#fff
    style OUT fill:#2D6A4F,color:#fff
```

#### Code walkthrough

```python
# 1. Define prompt — {topic} and {style} are filled at runtime
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a fun Python educator."),
    ("human",  "Explain {topic} in a {style} way."),
])

# 2. Build chain — data flows left to right through the pipe
chain = prompt | llm | StrOutputParser()

# 3. Invoke — pass a dict matching your {variables}
result = chain.invoke({"topic": "antigravity", "style": "fun"})

# 4. Reuse the same chain with completely different inputs
result2 = chain.invoke({"topic": "PEP 8", "style": "formal"})
```

**Key learning:** One chain definition = infinite reuse with different inputs.
Swap the LLM in one place, the whole app updates.

---

### Use Case 2 — Conversation with Memory

**File:** `use_cases/02_conversation_memory.py`
**Concepts:** `ConversationBufferMemory` · `ConversationChain` · `MessagesPlaceholder`

#### What it does

Builds a multi-turn chatbot called PythonBot that remembers everything
said in the conversation. Demonstrates the statelessness problem and
how LangChain memory solves it.

#### The Problem Without Memory

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant L as LLM (no memory)

    U->>L: "What is import antigravity?"
    L->>U: "It opens xkcd.com/353..."
    U->>L: "Who created it?"
    L->>U: "Who created what? I have no context."
```

#### The Solution With Memory

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant M as ConversationBufferMemory
    participant L as LLM (with memory)

    U->>L: "What is import antigravity?"
    L->>U: "It opens xkcd.com/353..."
    L->>M: Save: Human asked about antigravity, AI explained it

    U->>L: "Who created it?"
    M->>L: Load history: [Human: antigravity, AI: xkcd...]
    L->>U: "Barry Warsaw created it in 2008"
    L->>M: Save this exchange too
```

#### How MessagesPlaceholder Works

```mermaid
graph TD
    MEM[(Memory\nHumanMessage × 3\nAIMessage × 3)] -->|load_memory_variables| MP

    subgraph "Prompt Template"
        SYS["system: You are PythonBot..."]
        MP["MessagesPlaceholder\nvariable_name='history'\n← full history injected here"]
        HUM["human: {input}"]
    end

    MP --> FULL["Complete prompt sent to LLM:\nsystem + 6 history messages + new question"]
    FULL --> LLM[Groq LLM]
    LLM --> RESP[Response]
    RESP -->|save_context| MEM

    style MEM fill:#7D6608,color:#fff
    style LLM fill:#1B4F72,color:#fff
    style RESP fill:#2D6A4F,color:#fff
```

**Key learning:** `ConversationChain` automatically loads history before and
saves it after every call. You never manage the transcript manually.

---

### Use Case 3 — RAG Pipeline

**File:** `use_cases/03_rag_pipeline.py`
**Concepts:** `TextLoader` · `RecursiveCharacterTextSplitter` · `HuggingFaceEmbeddings` · `FAISS` · `RunnablePassthrough`

#### What it does

Builds a complete Retrieval Augmented Generation pipeline. Loads
`docs/python_knowledge.txt`, splits it, converts to vectors, stores
in FAISS, and answers questions using only the document — not the
LLM's training data.

#### Why RAG?

```
Without RAG:  "What is our internal refund policy?"
              → LLM: "I don't have access to your documents"

With RAG:     "What is our internal refund policy?"
              → Search your docs → find relevant paragraph
              → LLM reads paragraph → accurate answer
```

#### Full 7-Step Pipeline

```mermaid
graph TD
    DOC["📄 docs/python_knowledge.txt\n(your document)"] --> LOAD

    LOAD["1️⃣ TextLoader\nLoads file as Document objects\nDocument has: .page_content + .metadata"] --> SPLIT

    SPLIT["2️⃣ RecursiveCharacterTextSplitter\nchunk_size=500 chars\nchunk_overlap=50 chars\nSplits on: paragraph → line → word"] --> EMBED

    EMBED["3️⃣ HuggingFaceEmbeddings\nall-MiniLM-L6-v2 model (free, local)\nConverts each chunk → 384 numbers\n'Similar text = vectors close together'"] --> STORE

    STORE[("4️⃣ FAISS Vector Store\nStores all chunk vectors\nEnables millisecond similarity search")] --> RET

    Q(["❓ 'What is import antigravity?'"]) --> QEMB["Embed question → vector"]
    QEMB --> RET

    RET["5️⃣ Retriever\nFinds top-3 most similar chunks\nby cosine similarity"] --> FMT

    FMT["6️⃣ format_docs()\nJoins chunks into\none context string"] --> PROMPT

    Q --> PASS["RunnablePassthrough()\nPasses question unchanged"] --> PROMPT

    PROMPT["7️⃣ RAG Prompt\nsystem: Answer using ONLY this context\ncontext: {retrieved chunks}\nquestion: {user question}"] --> LLM

    LLM["Groq LLM\nAnswers using context\nnot its training data"] --> ANS(["✅ Accurate answer\ngrounded in your document"])

    style DOC fill:#4A4A4A,color:#fff
    style STORE fill:#922B21,color:#fff
    style Q fill:#2D6A4F,color:#fff
    style ANS fill:#2D6A4F,color:#fff
    style LLM fill:#1B4F72,color:#fff
    style PROMPT fill:#6C3483,color:#fff
```

#### The RAG Chain in Code

```python
rag_chain = (
    {
        "context":  retriever | format_docs,  # question → search → top chunks
        "question": RunnablePassthrough(),    # question → prompt unchanged
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is import antigravity?")
```

**Key learning:** RAG = your documents + LLM reasoning, without fine-tuning.
This pattern powers most enterprise LLM products including legal Q&A,
HR bots, internal knowledge bases, and customer support systems.

---

### Use Case 4 — Tools and Agents

**File:** `use_cases/04_tools_agent.py`
**Concepts:** `@tool` decorator · `create_react_agent` · `AgentExecutor` · ReAct loop

#### What it does

Builds a ReAct agent with 3 custom Python tools. The LLM dynamically
decides which tools to call and in what order to answer the question.

#### Chains vs Agents

```mermaid
graph LR
    subgraph "Chain — Fixed Sequence"
        CA[Input] --> CB[Step 1] --> CC[Step 2] --> CD[Output]
    end

    subgraph "Agent — Dynamic Decisions"
        AA[Input] --> AB{LLM decides}
        AB -->|needs tool A| AC[Tool A]
        AB -->|needs tool B| AD[Tool B]
        AC --> AB
        AD --> AB
        AB -->|done| AE[Final Answer]
    end

    style CB fill:#6C3483,color:#fff
    style CC fill:#6C3483,color:#fff
    style AB fill:#1B4F72,color:#fff
    style AC fill:#922B21,color:#fff
    style AD fill:#922B21,color:#fff
```

#### The 3 Custom Tools

```mermaid
graph TD
    AGENT[🤖 ReAct Agent] --> T1
    AGENT --> T2
    AGENT --> T3

    T1["🐍 get_python_easter_eggs(topic)\nReturns info about antigravity,\nimport this, __hello__\n\nDocstring tells LLM when to use it"]
    T2["🔢 calculate_python_stats(expression)\nEvaluates safe arithmetic\n'15 * 8' → 120\n\nSafe eval, no dangerous ops"]
    T3["📋 get_python_version_info(version)\nReturns Python 3.10/3.11/3.12\nfeatures and release notes"]

    style AGENT fill:#1B4F72,color:#fff
    style T1 fill:#922B21,color:#fff
    style T2 fill:#7D6608,color:#fff
    style T3 fill:#2D6A4F,color:#fff
```

#### Full ReAct Loop

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant E as AgentExecutor
    participant L as Groq LLM
    participant T as 🔧 Tools

    U->>E: "What is antigravity and what is 15 * 8?"
    E->>L: Question + tool descriptions
    L->>E: Thought: need easter egg info
    L->>E: Action: get_python_easter_eggs
    L->>E: Action Input: "antigravity"
    E->>T: Call tool
    T->>E: "Opens xkcd.com/353, Barry Warsaw 2008..."
    E->>L: Observation: tool result
    L->>E: Thought: now need calculation
    L->>E: Action: calculate_python_stats
    L->>E: Action Input: "15 * 8"
    E->>T: Call tool
    T->>E: "Result = 120"
    E->>L: Observation: 120
    L->>E: Thought: I have both answers
    L->>E: Final Answer: antigravity opens... 15*8=120
    E->>U: ✅ Complete answer
```

**Key learning:** Set `verbose=True` in `AgentExecutor` to see every
Thought/Action/Observation step printed in your terminal.
This is the best way to understand how agents reason.

---

### Use Case 5 — Streaming and Structured Output

**File:** `use_cases/05_antigravity_showcase.py`
**Concepts:** `.stream()` · `JsonOutputParser` · `.batch()`

#### What it does

Demonstrates three advanced LangChain output patterns using the antigravity
Easter egg as the topic: streaming tokens as they generate, forcing structured
JSON output, and processing multiple inputs in parallel.

#### Three Output Patterns

```mermaid
graph TD
    CHAIN[LCEL Chain\nprompt | llm | parser] --> INV
    CHAIN --> STR
    CHAIN --> BAT

    INV["invoke(input)\n\nWaits for full response\nReturns complete string\nUse for: simple one-off calls"]

    STR["stream(input)\n\nYields tokens one by one\nPrint as they arrive\nUse for: chatbots, real-time UI"]

    BAT["batch([input1, input2, input3])\n\nRuns all inputs in parallel\nFar faster than sequential invoke\nUse for: bulk processing"]

    style CHAIN fill:#1B4F72,color:#fff
    style INV fill:#4A4A4A,color:#fff
    style STR fill:#6C3483,color:#fff
    style BAT fill:#2D6A4F,color:#fff
```

#### Streaming — How It Works

```python
# Without streaming — user waits, then sees everything at once
result = chain.invoke({"question": "Tell me about antigravity"})
print(result)   # prints after full generation

# With streaming — user sees tokens appear immediately
for token in chain.stream({"question": "Tell me about antigravity"}):
    print(token, end="", flush=True)   # each token prints as it arrives
```

#### JsonOutputParser — Structured Data Extraction

```mermaid
graph LR
    PROMPT["System: Respond ONLY with valid JSON\nin this exact format:\n{easter_eggs: [{name, command, description}]}"] --> LLM
    LLM["Groq LLM"] --> RAW["Raw JSON string from LLM\n'{\"easter_eggs\": [{...}]}'"]
    RAW --> JOP["JsonOutputParser\nParses string → Python dict"]
    JOP --> OUT["Python dict ready to use\nresult['easter_eggs'][0]['name']"]

    style LLM fill:#1B4F72,color:#fff
    style JOP fill:#6C3483,color:#fff
    style OUT fill:#2D6A4F,color:#fff
```

**Key learning:** `.invoke()` for single calls, `.stream()` for real-time UX,
`.batch()` for throughput. All three work on any LCEL chain with zero
other changes needed.

---

## How to Run

Make sure `(venv)` appears in your terminal prompt before running.

```bash
# Show the menu
python main.py

# Run a specific use case
python main.py 1    # Simple chain
python main.py 2    # Memory chatbot
python main.py 3    # RAG pipeline
python main.py 4    # Agent with tools (set verbose=True to see reasoning)
python main.py 5    # Streaming + JSON

# Run all use cases in order
python main.py all

# Deactivate venv when done
deactivate
```

---

## Tech Stack

| Tool | Purpose | Cost |
|---|---|---|
| Python 3.9+ | Language | Free |
| LangChain | LLM framework (chains, memory, agents) | Free — open source |
| Groq | LLM inference API | Free tier available |
| Llama 3 8B | The LLM model (via Groq) | Free via Groq |
| FAISS | Vector similarity search | Free — Meta open source |
| HuggingFace `all-MiniLM-L6-v2` | Text embeddings (local) | Free — runs on CPU |
| python-dotenv | Load `.env` files safely | Free |

> **Total cost to run this project: $0**

---

## Key Learnings

After working through all 5 use cases you will understand:

| # | Concept | Taught In |
|---|---|---|
| 1 | LCEL `\|` pipe — composing chains | Use Case 1 |
| 2 | Prompt engineering — system vs human messages | Use Cases 1, 2 |
| 3 | Memory — making stateless LLMs stateful | Use Case 2 |
| 4 | RAG — LLMs + your own documents | Use Case 3 |
| 5 | Embeddings and vector search | Use Case 3 |
| 6 | Agents — LLM as a decision-maker | Use Case 4 |
| 7 | Tool design — `@tool` and docstring importance | Use Case 4 |
| 8 | Streaming — real-time token output | Use Case 5 |
| 9 | Structured output — JSON from LLMs | Use Case 5 |
| 10 | Project structure — venv, config, secrets management | All |

---

## Author

**Sneha**

[![GitHub](https://img.shields.io/badge/GitHub-snehas13-181717?style=flat&logo=github)](https://github.com/snehas13)

---

## License

MIT License — free to use, modify, and distribute.
