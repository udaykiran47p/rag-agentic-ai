import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def get_or_create_index(pc: Pinecone, index_name: str) -> None:
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' (dimension={EMBEDDING_DIMENSION}, metric='cosine')...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("Index created and ready.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")

def run_ingestion(pdf_path: str, index_name: str = PINECONE_INDEX_NAME) -> PineconeVectorStore:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"1. Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} raw pages.")

    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} text chunks.")

    print("3. Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    get_or_create_index(pc, index_name)

    print("4. Generating embeddings locally (on CPU) and storing in Pinecone...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("Ingestion completed successfully.")
    return vector_store

if __name__ == "__main__":
    target_pdf = os.path.join("data", "Ebook-Agentic-AI.pdf")
    run_ingestion(pdf_path=target_pdf)