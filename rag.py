"""
rag.py — A minimal, dependency-light RAG (Retrieval-Augmented Generation) pipeline.

Pipeline stages, in order:
  1. LOAD    -> read text out of a PDF or .txt file
  2. CHUNK   -> split the text into overlapping windows
  3. INDEX   -> turn each chunk into a vector (TF-IDF here) and store it
  4. RETRIEVE-> given a question, find the most similar chunks
  5. GENERATE-> stuff those chunks into a prompt and ask an LLM to answer

This version uses TF-IDF (scikit-learn) for retrieval instead of neural
embeddings. That's a deliberate choice for a first version: it has zero
heavy dependencies (no torch, no downloading model weights), it's fast,
and it's genuinely how retrieval was done for years before embedding
models took over. The "upgrade path" to real embeddings is explained
in the README and is a great v2 for your resume project.
"""

from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader


# ---------- 1. LOAD ----------

def load_text(path: str) -> str:
    """Read a .pdf or .txt file and return its raw text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


# ---------- 2. CHUNK ----------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks of `chunk_size` characters.

    Why overlap? If a sentence gets cut in half at a chunk boundary,
    overlap increases the odds that the full idea still appears intact
    in at least one chunk.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ---------- 3 & 4. INDEX + RETRIEVE ----------

@dataclass
class RagIndex:
    chunks: list[str]
    vectorizer: TfidfVectorizer
    matrix: any  # sparse TF-IDF matrix, one row per chunk


def build_index(chunks: list[str]) -> RagIndex:
    """Fit a TF-IDF vectorizer over the chunks and store the resulting matrix."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return RagIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def retrieve(index: RagIndex, question: str, top_k: int = 3) -> list[str]:
    """
    Return the top_k chunks most similar to the question.

    We vectorize the question with the SAME fitted vectorizer, then
    compute cosine similarity between the question vector and every
    chunk vector. Highest similarity = most relevant chunk.
    """
    q_vec = index.vectorizer.transform([question])
    sims = cosine_similarity(q_vec, index.matrix)[0]
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [index.chunks[i] for i in ranked_idx if sims[i] > 0]


# ---------- 5. GENERATE ----------

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, retrieved_chunks: list[str], client) -> str:
    """
    Send the question + retrieved context to Gemini and return the answer.
    `client` is a google.genai.Client() instance, passed in so this
    function stays easy to test without needing an API key up front.
    """
    prompt = build_prompt(question, retrieved_chunks)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text