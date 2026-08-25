# RAG Document Q&A

Ask questions about your own documents and get answers grounded in their actual
content — not the model's guesses. This is a minimal, from-scratch implementation
of Retrieval-Augmented Generation (RAG), built without LangChain/LlamaIndex so
every step is visible and understandable.

## How it works

```
your PDF/txt
     │
     ▼
 ┌─────────┐    ┌───────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
 │  LOAD   │───▶│ CHUNK │───▶│  INDEX   │───▶│ RETRIEVE  │───▶│ GENERATE │
 └─────────┘    └───────┘    └──────────┘    └───────────┘    └──────────┘
  extract         split       vectorize        find top-k       ask Claude
  raw text        into        chunks with       chunks most      using only
                  overlapping TF-IDF            similar to        those chunks
                  windows                       the question      as context
```

1. **Load** — `pypdf` pulls raw text out of a PDF (or we just read a `.txt`).
2. **Chunk** — the text is split into ~800-character overlapping windows.
   Overlap matters: if a fact gets cut off at a chunk boundary, the overlap
   means it still appears whole in a neighboring chunk.
3. **Index** — each chunk is converted into a TF-IDF vector (a numeric
   representation of which words matter most in that chunk, relative to the
   rest of the document).
4. **Retrieve** — the user's question is vectorized the same way, and we use
   cosine similarity to find the chunks whose vectors are closest to the
   question's vector.
5. **Generate** — the retrieved chunks are inserted into a prompt template and
   sent to Gemini, instructed to answer using *only* that context. This is
   what prevents hallucination — the model isn't answering from memory, it's
   answering from what you handed it.

### Why TF-IDF instead of neural embeddings?

Most RAG tutorials jump straight to embedding models (OpenAI's
`text-embedding-3-small`, or `sentence-transformers` running locally). Those
are more powerful — they understand meaning, not just word overlap — but they
add a large dependency (`torch`) and, if run locally, need to download model
weights.

TF-IDF is a legitimate retrieval method in its own right (it's how search
engines worked for years before embeddings), it has zero heavy dependencies,
and it makes the *mechanics* of retrieval easy to inspect and reason about.
Once this works, swapping in real embeddings is a small, well-scoped upgrade
(see "Next steps" below) — a great v2 to mention as "iterated on" in an
interview.

## Files

- `rag.py` — the actual pipeline: load, chunk, index, retrieve, generate.
- `app.py` — Streamlit UI wrapping the pipeline.
- `requirements.txt` — dependencies.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser tab that opens:
1. Paste your Gemini API key in the sidebar (get one free, no card required, at aistudio.google.com).
2. Upload a PDF or .txt file.
3. Ask a question.
4. Expand "Show retrieved chunks" to see exactly what the model based its answer on.

## Next steps to level this up

- **Swap TF-IDF for real embeddings**: use `sentence-transformers` (local,
  free) or OpenAI's embedding API, store vectors in a proper vector DB like
  ChromaDB or Pinecone instead of an in-memory matrix. This handles synonyms
  and meaning, not just matching words.
- **Multi-document support**: index a whole folder of files, not just one.
- **Citations**: have the model cite which chunk it used for each claim.
- **Conversation memory**: let follow-up questions reference earlier turns.

