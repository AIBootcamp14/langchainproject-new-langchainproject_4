# src/schemas/rag_schema.py

from pydantic import BaseModel, Field
from typing import List

# 사용자 요청 데이터 형식
class QuestionRequest(BaseModel):
    """
    API로 들어오는 사용자 질문 요청 스키마
    """
    question: str = Field(..., description="사용자가 챗봇에게 던지는 질문")

# API 응답 데이터 형식
class AnswerResponse(BaseModel):
    """
    RAG 챗봇의 답변 및 참조 출처를 포함하는 응답 스키마
    """
    answer: str = Field(..., description="LLM이 생성한 최종 답변")
    sources: List[str] = Field(..., description="답변에 사용된 문서의 출처 URL 리스트")