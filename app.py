"""
app.py — Streamlit front-end for the RAG pipeline in rag.py.

Run with:  streamlit run app.py

Flow:
  1. User uploads a PDF or txt file, and enters their Anthropic API key.
  2. We load + chunk + index the file (cached so it doesn't redo work).
  3. User types a question.
  4. We retrieve the most relevant chunks and ask Claude to answer using them.
  5. We show the answer AND which chunks it was based on (for transparency —
     this is what makes RAG trustworthy instead of a black box).
"""

import streamlit as st
from google import genai
from rag import load_text, chunk_text, build_index, retrieve, generate_answer

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Chat with your document")
st.caption("Retrieval-Augmented Generation demo — ask questions, get answers grounded in your own file.")

with st.sidebar:
    api_key = st.text_input("Gemini API key", type="password")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    top_k = st.slider("Chunks to retrieve per question", 1, 5, 3)

if uploaded_file and api_key:
    # Cache the index so re-asking questions doesn't re-process the file.
    cache_key = uploaded_file.name
    if st.session_state.get("cache_key") != cache_key:
        with open(f"/tmp/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getvalue())
        text = load_text(f"/tmp/{uploaded_file.name}")
        chunks = chunk_text(text)
        st.session_state["index"] = build_index(chunks)
        st.session_state["cache_key"] = cache_key
        st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

    question = st.text_input("Ask a question about the document")
    if question:
        client = genai.Client(api_key=api_key)
        index = st.session_state["index"]
        retrieved = retrieve(index, question, top_k=top_k)

        if not retrieved:
            st.warning("No relevant content found in the document for that question.")
        else:
            with st.spinner("Thinking..."):
                answer = generate_answer(question, retrieved, client)
            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Show retrieved chunks (what the answer was based on)"):
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)
else:
    st.info("Enter your API key and upload a document to get started.")