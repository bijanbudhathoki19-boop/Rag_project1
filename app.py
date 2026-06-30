
import streamlit as st
from pathlib import Path
from utils import load_pdf, chunk_text
from rag_engine import RAGEngine

st.set_page_config(page_title="Bijan AI RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Bijan AI RAG Chatbot")
st.caption("Production-ready modular RAG chatbot")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = {}

with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        for pdf in uploaded:
            if pdf.name not in st.session_state.parsed_docs:
                save_path = Path("uploads")
                save_path.mkdir(exist_ok=True)
                file_path = save_path / pdf.name
                file_path.write_bytes(pdf.read())

                text = load_pdf(str(file_path))
                chunks = chunk_text(text)

                st.session_state.rag.add_chunks(chunks)

                st.session_state.parsed_docs[pdf.name] = {
                    "text": text,
                    "chunks": chunks
                }

    st.divider()
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

        results = st.session_state.rag.search(prompt, top_k=top_k)

        if results:
            context = "\n\n".join([r[0] for r in results])
            answer = f"Retrieved Context:\n\n{context}"
        else:
            answer = "No relevant information found in the uploaded PDFs."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

with tab_docs:
    if not st.session_state.parsed_docs:
        st.info("Upload PDFs to see parsed text and chunks.")
    else:
        for name, data in st.session_state.parsed_docs.items():
            st.subheader(name)
            st.text_area(
                "Parsed Text",
                data["text"][:3000],
                height=200,
                key=f"text_{name}"
            )

            st.write(f"Chunks: {len(data['chunks'])}")
            for i, c in enumerate(data["chunks"][:10]):
                with st.expander(f"Chunk {i+1}"):
                    st.write(c)

with tab_pipeline:
    st.markdown("""
1. Upload PDF ✅
2. Parse PDF (PyPDF2/LlamaParse) ✅
3. Chunk Text ✅
4. Generate Embeddings ✅
5. FAISS Search ✅
6. Retrieve Chunks ✅
7. Send to LLM
8. Final Answer
""")
