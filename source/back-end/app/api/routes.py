from fastapi import APIRouter, HTTPException
from app.api.schemas import QuestionRequest, AnswerResponse
from app.services.rag_pipeline import generate_answer

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    try:
        answer = generate_answer(payload.question)
        return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=502, detail="Error processing request")