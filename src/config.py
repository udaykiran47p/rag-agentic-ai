import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "agentic-ai-local-index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # MiniLM uses 384 dimensions
LLM_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 4

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment or .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment or .env file.")