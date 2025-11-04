# src/schemas/rag_schema.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 사용자 요청 스키마 ---
class QuestionRequest(BaseModel):
    """
    API로 들어오는 사용자 질문 요청 스키마 (main.py의 Question 모델과 일치)
    """
    question: str = Field(..., description="사용자의 질문 내용", min_length=5, max_length=500)
    
    # 💡 [추가] 대화형 메모리 기능을 위해 session_id가 필수입니다.
    session_id: Optional[str] = Field("default_session", description="사용자 세션 ID (기억/히스토리 기능용)", max_length=50)


# --- API 응답 스키마 ---
class SourceModel(BaseModel):
    """
    답변의 근거가 되는 출처 문서 스키마
    """
    url: Optional[str] = Field(None, description="문서의 원본 URL")
    title: Optional[str] = Field(None, description="문서의 제목")

class AnswerResponse(BaseModel):
    """
    RAG 챗봇의 답변 및 참조 출처를 포함하는 응답 스키마 (main.py의 Answer 모델과 일치)
    """
    answer: str = Field(..., description="LLM이 생성한 최종 답변")
    
    # 💡 [수정] 단순 List[str]이 아닌, 더 풍부한 정보를 제공하는 SourceModel 리스트로 변경
    sources: List[SourceModel] = Field(..., description="답변에 사용된 출처 문서 목록 (url, title)")
    
    execution_time_ms: Optional[float] = Field(None, description="질문 처리 실행 시간 (밀리초)")