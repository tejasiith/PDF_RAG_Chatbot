import streamlit as st
from rag_chain import get_rag_chain

st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot")

# Load RAG pipeline
rag_pipeline = get_rag_chain()

# Chat interface
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about your PDFs:")

if query:
    answer = rag_pipeline(query)
    st.session_state.history.append((query, answer))

# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**Your Question:** {q}")
    st.markdown(f"**Answer:** {a}")
    st.markdown("---")