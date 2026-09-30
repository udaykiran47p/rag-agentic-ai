import streamlit as st
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

st.set_page_config(page_title="Agentic AI Assistant", layout="wide")

st.title("📖 Agentic AI Knowledge Assistant")
st.caption("Grounded Question Answering powered by LangGraph & Pinecone")

@st.cache_resource
def get_graph():
    return build_rag_graph(index_name=PINECONE_INDEX_NAME)

graph = get_graph()

user_query = st.text_input("Ask a question about the Agentic AI eBook:", placeholder="e.g., What are the core components of an Agentic Architecture?")

if st.button("Submit Query", type="primary") and user_query:
    with st.spinner("Retrieving knowledge and formulating response..."):
        state = {
            "question": user_query,
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        res = graph.invoke(state)

    col1, col2 = st.columns()

    with col1:
        st.subheader("Answer")
        st.write(res["answer"])

    with col2:
        st.subheader("Metadata")
        st.metric("Confidence Score", f"{res['score']:.2f}")
        
        with st.expander("Retrieved Context Chunks", expanded=False):
            for i, chunk in enumerate(res["context"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.info(chunk)