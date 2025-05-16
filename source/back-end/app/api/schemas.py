from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Texto de la pregunta del usuario")

class AnswerResponse(BaseModel):
    answer: str