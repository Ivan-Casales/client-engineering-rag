from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.api.schemas import QuestionRequest, AnswerResponse, ChatRequest, ChatResponse
from app.services.rag.rag_pipeline import generate_answer
from app.services.vectorstore.loader_service import process_pdf_upload
from langchain.chains import RetrievalQA
from app.services.rag.chat_service import process_chat
from app.services.container import rag_chain, reranker

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    try:
        answer = generate_answer(payload.question, rag_chain, reranker)
        return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error processing request: {e}")

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text chunks, and index them into the vector store.
    """
    file_bytes = await file.read()
    num_chunks, error = process_pdf_upload(file_bytes)

    if error:
        return JSONResponse(status_code=500, content={"detail": f"Failed to process file: {error}"})
    
    return {"detail": f"{num_chunks} chunks indexed successfully."}

@router.post("/chat", response_model=ChatResponse)
async def chat_conversation(payload: ChatRequest):
    """
    Chat endpoint that maintains context across turns.
    """
    try:
        new_history, answer = process_chat(payload.message, payload.history)
        return ChatResponse(answer=answer, history=new_history)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))