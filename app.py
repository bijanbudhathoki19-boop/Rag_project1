"""
app.py
Starter skeleton for the modular RAG Chatbot.

This file is intentionally a scaffold because the complete production version
is too large to fit in a single ChatGPT response. It is designed to integrate
with:
    - rag_engine.py
    - ui_components.py
    - utils.py
    - config.py
"""

import streamlit as st

st.set_page_config(
    page_title="Bijan AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Bijan AI RAG Chatbot")
st.caption("Production-ready modular RAG chatbot (starter scaffold)")

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("⚙ Settings")
    theme = st.selectbox("Theme", ["Dark", "Light", "Ocean", "Purple"])
    top_k = st.slider("Top K", 1, 10, 5)

tab_chat, tab_docs, tab_pipeline = st.tabs(
    ["💬 Chat", "📚 Documents", "🔬 RAG Pipeline"]
)

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = (
            "Placeholder response. Connect this to rag_engine.answer_query() "
            "after generating the remaining project files."
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

with tab_docs:
    st.info("Parsed PDFs, chunks, and metadata viewer will be added here.")

with tab_pipeline:
    st.markdown("""
1. Upload PDF
2. Parse (LlamaParse)
3. Chunk
4. Embed
5. FAISS Search
6. Retrieve
7. Groq LLM
8. Final Answer
""")
