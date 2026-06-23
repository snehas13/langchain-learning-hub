"""
USE CASE 3 — RAG (Retrieval Augmented Generation)
==================================================
CONCEPTS:
  Document Loading     : Load text files as LangChain Document objects
  Text Splitting       : Break large docs into chunks (LLMs have token limits)
  Embeddings           : Convert text → numbers (vectors) that encode meaning
  Vector Store (FAISS) : Store + search vectors by semantic similarity
  Retriever            : Finds the top-k most relevant chunks for a query
  RAG Chain            : Retriever → inject chunks into prompt → LLM answers

WHY RAG?
  LLMs are trained up to a cutoff date and don't know YOUR data.
  RAG lets you ask questions about any document without retraining.

FLOW:
  Question → Embed question → Search vector store → Get relevant chunks
  → Inject chunks into prompt → LLM answers using chunks as context
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.config import get_llm, banner, info, success


def run():
    banner("Use Case 3 — RAG Pipeline (Search Your Docs)", "green")

    # STEP 1 — Load documents
    # TextLoader reads a .txt file and wraps it in a Document object
    # Document has: .page_content (the text) and .metadata (source info)
    info("Step 1 → Loading document from docs/python_knowledge.txt")
    loader = TextLoader("docs/python_knowledge.txt", encoding="utf-8")
    documents = loader.load()
    print(f"  Loaded {len(documents)} document(s)")

    # STEP 2 — Split into chunks
    # LLMs have token limits. A 10,000 word doc must be split into chunks.
    # RecursiveCharacterTextSplitter tries to split on: \n\n → \n → space
    # chunk_size    = max characters per chunk
    # chunk_overlap = how much adjacent chunks share (preserves context at edges)
    info("Step 2 → Splitting into chunks (chunk_size=500, overlap=50)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks from the document")

    # STEP 3 — Create embeddings
    # Embeddings convert text → a list of numbers (vector) that represents meaning
    # Similar meaning = vectors close together in space
    # We use HuggingFace (free, runs locally — no API key needed for embeddings!)
    info("Step 3 → Creating embeddings (HuggingFace, runs locally — free!)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # STEP 4 — Build vector store
    # FAISS is Facebook's fast similarity search library
    # It stores each chunk's vector and lets us search by semantic similarity
    info("Step 4 → Building FAISS vector store from chunks")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("  Vector store built and ready!")

    # STEP 5 — Create retriever
    # Retriever wraps the vector store with a simple .invoke(query) interface
    # search_kwargs={"k": 3} → return top 3 most similar chunks
    info("Step 5 → Creating retriever (top-3 chunks per query)")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # STEP 6 — Build RAG prompt
    # {context} = the retrieved chunks (injected by the chain)
    # {question} = the user's question
    info("Step 6 → Building RAG prompt with {context} and {question}")
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Python knowledge assistant. "
            "Answer questions using ONLY the context provided below. "
            "If the context doesn't contain the answer, say 'I don't know based on this document.'\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}"),
    ])

    # STEP 7 — Build RAG chain using LCEL
    # RunnablePassthrough() passes the question through unchanged
    # The chain:
    #   1. Takes {"question": "..."} as input
    #   2. Retriever finds relevant chunks → formatted as string → {context}
    #   3. RunnablePassthrough passes question → {question}
    #   4. rag_prompt formats the full prompt
    #   5. LLM generates answer
    #   6. Parser extracts string
    info("Step 7 → Building RAG chain (retriever | prompt | llm | parser)")

    def format_docs(docs):
        """Join retrieved chunks into a single string for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context":  retriever | format_docs,  # retrieve + format
            "question": RunnablePassthrough(),     # pass question as-is
        }
        | rag_prompt
        | get_llm(temperature=0.3)  # lower temp = more factual
        | StrOutputParser()
    )

    # STEP 8 — Ask questions!
    info("Step 8 → Asking questions about the document\n")
    questions = [
        "What is the import antigravity Easter egg?",
        "What is the Zen of Python?",
        "How does Python handle errors according to this document?",
    ]

    for q in questions:
        print(f"  ❓ {q}")
        answer = rag_chain.invoke(q)
        print(f"  💡 {answer}\n")

    success("Lesson: RAG = your documents + LLM reasoning, without fine-tuning!")


if __name__ == "__main__":
    run()