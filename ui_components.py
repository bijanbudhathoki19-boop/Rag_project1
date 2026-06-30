import streamlit as st
from rag_engine import RAGEngine

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS (Chat UI Styling)
# -----------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 8px 0;
    text-align: right;
}

.bot-msg {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 8px 0;
    text-align: left;
}

.title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize RAG Engine
# -----------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "RAG Chatbot using FAISS + Groq/Gemini LLM + Document Retrieval"
)

st.sidebar.markdown("### Features")
st.sidebar.write("• PDF Knowledge Base")
st.sidebar.write("• FAISS Vector Search")
st.sidebar.write("• LLM Response Generation")
st.sidebar.write("• Chat Memory")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<div class='title'>💬 RAG Chatbot</div>", unsafe_allow_html=True)

# Chat history display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-msg'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-msg'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True
        )

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask anything about your documents...")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get response from RAG engine
    response = st.session_state.rag.generate_response(
        query=user_input,
        history=st.session_state.messages
    )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()