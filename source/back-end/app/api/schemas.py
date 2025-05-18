from pydantic import BaseModel, Field, field_validator
from typing import List, Dict

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question text")

    @field_validator('question')
    def validate_question(cls, v: str) -> str:
        if len(v) > 512:
            raise ValueError("Question must be at most 512 characters long")
        if '\n' in v or '\r' in v:
            raise ValueError("Newlines are not allowed in the question")
        return v.strip()

class AnswerResponse(BaseModel):
    answer: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of prior turns as dicts with keys 'user' and 'assistant'"
    )

    @field_validator('message')
    def validate_message(cls, v: str) -> str:
        if len(v) > 512:
            raise ValueError("Message must be at most 512 characters long")
        if '\n' in v or '\r' in v:
            raise ValueError("Newlines are not allowed in the message")
        return v.strip()

class ChatResponse(BaseModel):
    answer: str
    history: List[Dict[str, str]]