from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question text")

class AnswerResponse(BaseModel):
    answer: str