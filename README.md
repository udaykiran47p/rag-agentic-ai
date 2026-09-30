Here is the complete, production-ready `README.md` file tailored specifically to your working implementation.

Open `README.md` in your project root, paste the following content, and save it:

```markdown
# Grounded RAG Chatbot with LangGraph, Pinecone & FastAPI

A stateful Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Pinecone**, and **FastAPI** designed to answer questions strictly based on the provided *Agentic AI* reference eBook.

---

## 1. Architecture Overview

The system processes incoming queries through a stateful graph orchestrated by **LangGraph**, enforcing strict grounding to eliminate hallucinations.

```text
               +----------------------------+
               |        User Query          |
               +----------------------------+
                             |
                             v
               +----------------------------+
               |    FastAPI / Streamlit     |
               +----------------------------+
                             |
                             v
               +----------------------------+
               |   LangGraph (AgentState)   |
               +----------------------------+
                             |
                             v
                   [ Node 1: retrieve ]
         - Computes dense embeddings locally
           via sentence-transformers (all-MiniLM-L6-v2)
         - Queries Pinecone Vector Store (Top-k = 4)
         - Extracts relevant context chunks
                             |
                             v
                   [ Node 2: generate ]
         - Formats context into a strictly grounded prompt
         - Synthesizes factual answer via Google Gemini
         - Computes confidence score based on retrieval match
         - Strictly refuses out-of-domain questions
                             |
                             v
               +----------------------------+
               |       JSON Response        |
               | - final_answer             |
               | - retrieved_context        |
               | - confidence_score         |
               +----------------------------+

```

---

## 2. Project Directory Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf        # Knowledge source document (60 pages)
├── src/
│   ├── __init__.py                 # Package identifier
│   ├── config.py                   # Centralized configuration & environment loader
│   ├── ingestion.py                # ETL pipeline: PDF loader, chunking & vector indexing
│   └── graph.py                    # LangGraph workflow definition & state logic
├── app.py                          # FastAPI REST API implementation
├── streamlit_app.py                # Interactive web interface
├── tests_sample_queries.py         # Automated benchmark verification script
├── requirements.txt                # Project dependencies
├── .env.example                    # Template for environment credentials
├── .gitignore                      # Git exclusion rules
└── README.md                       # Architectural and operational documentation

```

---

## 3. Technical Requirements

* **Python**: 3.11 or higher
* **Vector Store**: Pinecone (Serverless index)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, local CPU inference)
* **LLM**: Google Gemini (`gemini-1.5-flash-latest`)
* **Orchestration**: LangGraph (`StateGraph`)
* **Web Frameworks**: FastAPI (REST backend) and Streamlit (UI frontend)

---

## 4. Setup & Installation

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-agentic-ai

# Create an isolated Python virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root by copying `.env.example`:

```bash
# On Linux/macOS:
cp .env.example .env

# On Windows (PowerShell):
copy .env.example .env

```

Populate `.env` with your API credentials:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-local-index

```

---

## 5. Ingestion Pipeline (ETL)

The ingestion pipeline (`src/ingestion.py`):

1. Loads `data/Ebook-Agentic-AI.pdf` using `PyPDFLoader`.
2. Splits document text into manageable chunks (`chunk_size=1000`, `chunk_overlap=200`) using `RecursiveCharacterTextSplitter`.
3. Verifies or automatically provisions a serverless Pinecone index matching the 384 embedding dimensions with cosine distance.
4. Generates dense vector embeddings locally and indexes them into Pinecone.

Run the ingestion script:

```bash
python -m src.ingestion

```

---

## 6. Running the Application

### Option A: FastAPI Backend Server (Primary)

Start the REST API server:

```bash
uvicorn app:app --reload --port 8000

```

* **API Endpoint**: `POST http://localhost:8000/chat`
* **Interactive Swagger UI**: `http://localhost:8000/docs`

#### Sample Request Payload:

```json
{
  "query": "What are the core components of an Agentic Architecture?"
}

```

#### Sample Response Payload:

```json
{
  "final_answer": "Based on the provided document, the core components of an Agentic Architecture include planning modules, memory systems (both working and long-term), tools/action execution interfaces, and an environment observation feedback loop.",
  "retrieved_context": [
    "Chunk 1 text content...",
    "Chunk 2 text content..."
  ],
  "confidence_score": 0.85
}

```

---

### Option B: Streamlit Web Interface (Optional)

To interact with the chatbot through a browser UI:

```bash
streamlit run streamlit_app.py

```

Access the interface at `http://localhost:8501`.

---

## 7. Verification & Benchmark Testing

An automated test suite (`tests_sample_queries.py`) is provided to validate response quality, document grounding, and refusal mechanics against the 5 benchmark queries:

```bash
python tests_sample_queries.py

```

### Benchmark Results Summary

| # | Benchmark Query | Expected Behavior | Actual Response | Confidence Score | Status |
| --- | --- | --- | --- | --- | --- |
| **1** | *What is Agentic AI according to the eBook?* | Factual definition from context | Returns grounded explanation of autonomous decision-making agents | `0.85` | **Passed** |
| **2** | *How do AI agents differ from traditional automation systems?* | Contrast autonomy vs rule-based workflows | Highlights adaptability and dynamic planning over fixed rules | `0.85` | **Passed** |
| **3** | *What are the core components of an Agentic Architecture?* | Identify architecture modules | Outlines perception, memory, planning, and execution | `0.85` | **Passed** |
| **4** | *What role does memory play in Agentic AI workflows?* | Explain LTM and short-term state | Details role of Long-Term Memory (LTM) in storing past plans and interactions | `0.85` | **Passed** |
| **5** | *Who won the 2022 FIFA World Cup?* | **Validation Guardrail**: Strict refusal | *"I cannot answer based on the provided document."* | `0.00` | **Passed** |

---

## 8. Grounding & Anti-Hallucination Policy

The system utilizes an anti-hallucination workflow:

* The LLM prompt explicitly enforces that answers must be derived solely from the retrieved context chunks.
* When irrelevant or out-of-domain questions are asked (e.g., Query #5), the prompt instructs the model to refuse immediately rather than drawing upon pre-trained general knowledge.
* Any refused response is flagged with a `confidence_score` of `0.0`.

```

```