"""
LoanDesk AI - RAG Pipeline
Retrieval Augmented Generation over company loan documents.
Flow: ingest (chunk -> embed -> store) then answer (retrieve -> ground -> cite)
"""

import re

import anthropic
import chromadb
from dotenv import load_dotenv
from pathlib import Path

# ---------- SETUP ----------

# Read ANTHROPIC_API_KEY from a .env file in the project root, if present
load_dotenv()

# Persistent client: saves vectors to disk, so we ingest once and reuse
client = chromadb.PersistentClient(path="./loandesk_db")
collection = client.get_or_create_collection(name="loandesk")

# Anthropic client: reads ANTHROPIC_API_KEY from environment
llm = anthropic.Anthropic()


# ---------- INGESTION (offline, runs once) ----------

def split_sections(text):
    """Split a document at its structural boundaries: numbered policy headings
    ("3. INCOME VERIFICATION") or FAQ entries ("Q: ...").

    Returns (title, sections). Anything before the first heading is treated as
    the document title.
    """
    boundary = re.compile(r"^(?=\d+\.\s+[A-Z]|Q:\s)", re.M)
    parts = [p.strip() for p in boundary.split(text) if p.strip()]
    if parts and not re.match(r"^(\d+\.\s+[A-Z]|Q:\s)", parts[0]):
        return parts[0].split("\n")[0].strip(), parts[1:]
    return "", parts


def chunk_text(text, max_chars=900):
    """One chunk per section, never split mid-sentence.

    Fixed-width slicing cut rules in half — the OCR confidence rule ended at
    "...the document goes to the ", and one FAQ question was separated from its
    own answer. Sections here are 165-420 chars, so each fits whole. Each chunk
    is prefixed with the document title, since a bare "5. EXPIRED DOCUMENTS"
    does not say which policy it belongs to.
    """
    title, sections = split_sections(text)
    chunks = []
    for sec in sections or [text]:
        sec = f"{title}\n\n{sec}" if title else sec
        while len(sec) > max_chars:      # oversized section: fall back to slicing
            chunks.append(sec[:max_chars])
            sec = sec[max_chars:]
        if sec.strip():
            chunks.append(sec)
    return chunks


def ingest():
    """Load all txt docs, chunk them, store in the vector DB.
    ChromaDB embeds each chunk automatically (default model: all-MiniLM)."""
    docs, ids, metas = [], [], []
    files = list(Path("docs").glob("*.txt"))

    for file in files:
        text = file.read_text()
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{file.stem}_{i}")          # unique id: loan_policy_0
            metas.append({"source": file.name})     # metadata powers citations

    collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"Ingested {len(docs)} chunks from {len(files)} files")


# ---------- RETRIEVAL ONLY (debugging helper) ----------

def ask(question):
    """Show raw retrieved chunks with distances. Lower distance = more similar.
    Useful for debugging retrieval before blaming the LLM."""
    results = collection.query(query_texts=[question], n_results=3)
    print(f"\nQuestion: {question}\n")
    for doc, meta, dist in zip(results["documents"][0],
                               results["metadatas"][0],
                               results["distances"][0]):
        print(f"--- {meta['source']} (distance: {dist:.3f}) ---")
        print(doc[:250], "\n")


# ---------- FULL RAG: RETRIEVE + GENERATE WITH CITATIONS ----------

def answer(question):
    """Retrieve top chunks, ground the LLM in them, answer with citations."""
    results = collection.query(query_texts=[question], n_results=3)

    # Build context: each chunk labeled with its source file
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[{meta['source']}]\n{doc}\n\n"

    prompt = f"""Answer the question using ONLY the context below.
After each fact, mention the source file in brackets, like [loan_policy.txt].
If the context does not contain the answer, say exactly:
"I don't have this information in the company documents."

Context:
{context}

Question: {question}"""

    response = llm.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ---------- MAIN LOOP ----------

if __name__ == "__main__":
    # Ingest only on first run (persistent DB remembers)
    if collection.count() == 0:
        ingest()
    else:
        print(f"Using existing index: {collection.count()} chunks")

    print("LoanDesk AI ready. Type a question, 'raw' mode shows chunks, 'quit' exits.\n")
    mode = "answer"

    while True:
        q = input("Ask (or 'raw'/'answer'/'quit'): ").strip()
        if q == "quit":
            break
        elif q == "raw":
            mode = "raw"
            print("Raw retrieval mode ON\n")
        elif q == "answer":
            mode = "answer"
            print("Full answer mode ON\n")
        elif q:
            if mode == "raw":
                ask(q)
            else:
                print("\n" + answer(q) + "\n")
