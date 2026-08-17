"""
Test client for the router API.
Run server first (from project root): python -m api.app
Then run: python api/client.py
"""
import httpx

BASE_URL = "http://localhost:8000"


def ask_gemini(question: str) -> str:
    res = httpx.post(f"{BASE_URL}/gemini/invoke", json={"input": question}, timeout=60)
    res.raise_for_status()
    return res.json()["reply"]


def ask_ollama(question: str) -> str:
    res = httpx.post(f"{BASE_URL}/ollama/invoke", json={"input": question}, timeout=60)
    res.raise_for_status()
    return res.json()["reply"]


def ask_both(question: str) -> dict:
    res = httpx.post(f"{BASE_URL}/router/invoke", json={"input": question}, timeout=60)
    res.raise_for_status()
    return res.json()["responses"]


if __name__ == "__main__":
    question = "What is the capital of France?"

    print(f"Question: {question}")
    print("-" * 50)

    responses = ask_both(question)
    print(f"[Gemini]\n{responses['gemini']}")
    print(f"\n[Ollama]\n{responses['ollama']}")
