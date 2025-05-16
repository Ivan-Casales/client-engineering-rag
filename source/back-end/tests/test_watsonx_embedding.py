import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.watsonx_client import get_embedding

def test_embedding():
    print("TEST EMBEDDING")

    test_text = "What is Watsonx.ai?"
    try:
        embedding = get_embedding(test_text)
        print("Watsonx embedding retrieved successfully.")
        print(f"Length of embedding: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error generating embedding: {e}")

if __name__ == "__main__":
    test_embedding()
