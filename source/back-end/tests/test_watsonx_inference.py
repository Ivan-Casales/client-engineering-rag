import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.watsonx_client import generate_answer_with_context

def test_qa():
    print("TEST QA")
    context = (
        "Watsonx.ai is IBM’s next‐generation AI and data platform. "
        "Permite a las empresas entrenar, validar e implementar modelos de IA en producción."
    )
    question = "¿Para qué sirve Watsonx.ai?"
    try:
        answer = generate_answer_with_context(context, question)
        print("QA executed")
        print("Answer:", answer)
    except Exception as e:
        print(f"Error in QA: {e}")

if __name__ == "__main__":
    test_qa()
