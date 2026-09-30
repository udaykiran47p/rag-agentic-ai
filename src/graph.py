from typing import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

from src.config import (
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K_RETRIEVAL,
)

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str = PINECONE_INDEX_NAME):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)

    def retrieve_node(state: AgentState) -> dict:
        query = state.get("question", "")
        docs_with_scores = vector_store.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)
        
        context_chunks = [doc.page_content for doc, _ in docs_with_scores]
        
        if docs_with_scores:
            avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
            confidence = round(float(avg_score), 2)
        else:
            confidence = 0.0

        return {
            "context": context_chunks,
            "score": confidence,
        }

    def generate_node(state: AgentState) -> dict:
        question = state.get("question", "")
        context_list = state.get("context", [])

        context_str = "\n\n---\n\n".join(context_list)

        prompt = f"""You are an expert AI assistant answering questions about an Agentic AI eBook.
Answer the user's question using the provided context below. The context is extracted directly from the eBook.
If the context does not contain enough information to answer the question, state: 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)

        if isinstance(response.content, list):
            answer_text = "".join(
                item if isinstance(item, str) else item.get("text", "") 
                for item in response.content
            ).strip()
        else:
            answer_text = str(response.content).strip()

        if "cannot answer based on the provided document" in answer_text.lower() or not context_list:
            confidence = 0.0
        else:
            confidence = 0.85 if len(context_list) > 0 else 0.0

        return {
            "answer": answer_text,
            "score": confidence,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()