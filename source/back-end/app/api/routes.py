from fastapi import APIRouter
from app.api.schemas import QuestionRequest, AnswerResponse
from app.services.rag_pipeline import generate_answer

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    question = payload.question
    answer = generate_answer(question)
    return {"answer": answer}