import json
import requests

API_URL = "http://localhost:8000/chat"

BENCHMARK_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # Validation check: must refuse
]

def run_benchmarks():
    print("=" * 70)
    print("Starting Benchmark Verification Suite")
    print("=" * 70)

    for idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n[Test {idx}] Query: {query}")
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                data = response.json()
                print(f"Confidence Score : {data['confidence_score']}")
                print(f"Retrieved Chunks : {len(data['retrieved_context'])} chunks")
                print(f"Answer Preview   : {data['final_answer'][:250]}...")
            else:
                print(f"Failed with status code {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("Error: FastAPI server is not running on port 8000. Start it with 'uvicorn app:app --reload'.")
            break

if __name__ == "__main__":
    run_benchmarks()