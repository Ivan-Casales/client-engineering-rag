from pydantic import BaseModel, Field
from typing import List, Dict

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question text")

class AnswerResponse(BaseModel):
    answer: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of prior turns as dicts with keys 'user' and 'assistant'"
    )

class ChatResponse(BaseModel):
    answer: str
    history: List[Dict[str, str]]