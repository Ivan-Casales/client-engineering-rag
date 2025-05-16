import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.watsonx_client import rerank_documents

def test_rerank():
    print("TEST RERANK")
    question = "What is Watsonx.ai?"
    docs = [
        "Watsonx.ai is IBM's cloud AI and data platform.",
        "OpenAI develops models like GPT-4.",
        "Watsonx.ai ofrece herramientas de IA para empresas."
    ]
    try:
        ranking = rerank_documents(question, docs)
        print("Rerank executed")
        for i, doc in enumerate(ranking, 1):
            print(f"{i}. {doc}")
    except Exception as e:
        print(f"Error in rerank: {e}")

if __name__ == "__main__":
    test_rerank()
