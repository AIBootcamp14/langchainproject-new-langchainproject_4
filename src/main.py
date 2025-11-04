# src/main.py

"""
FastAPI 애플리케이션 정의 및 RAG API 엔드포인트
"""

import os
import time
from typing import Dict, Any, Optional

# 써드파티 라이브러리
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request # Request 추가
from pydantic import BaseModel, Field

# 프로젝트 모듈
from src.modules.retriever import RAGRetriever # RAGRetriever 임포트

# 환경 변수 미리 로드 (필요하다면)
load_dotenv()

# --- Pydantic 모델 정의 ---
# PEP 484: 타입 힌트와 기본값 명시
class QueryModel(BaseModel):
    """사용자 질문을 위한 입력 스키마"""
    question: str = Field(..., description="사용자의 RAG 질문")

class ResponseModel(BaseModel):
    """RAG 답변 및 메타데이터를 위한 출력 스키마"""
    answer: str = Field(..., description="RAG 챗봇이 생성한 답변")
    source_urls: Optional[List[str]] = Field(None, description="참조된 원본 문서 URL 리스트")
    execution_time_ms: int = Field(..., description="RAG 파이프라인 총 실행 시간 (밀리초)")

# --- FastAPI 앱 및 RAGRetriever 초기화 ---
app = FastAPI(
    title="LangChain Document RAG API",
    description="Upstage Solar LLM과 ChromaDB를 활용한 LangChain 문서 검색 증강 생성(RAG) API.",
    version="1.0.0",
)

# RAG Retriever 인스턴스를 저장할 변수 (초기화는 startup에서 진행)
rag_retriever: Optional[RAGRetriever] = None 


# 💡 [핵심 수정]: FastAPI의 Startup 이벤트를 활용하여 RAG 파이프라인 초기화
@app.on_event("startup")
async def startup_event():
    """
    FastAPI 서버 시작 시 RAGRetriever를 초기화하고 종속성을 확인합니다.
    """
    global rag_retriever
    print("\n--- FastAPI Startup: RAG 파이프라인 초기화 중 ---")
    try:
        # RAGRetriever 초기화 (LLM, 임베딩, DB 연결)
        rag_retriever = RAGRetriever()
        print("✅ RAGRetriever 초기화 성공")
        
        # 간단한 LLM/Embedding 연결 테스트 (src/modules/llm.py의 test_connection에 의존)
        if not rag_retriever.vdb_client.health_check():
             print("❌ 경고: ChromaDB 연결에 실패했습니다. /ask 엔드포인트 사용 불가.")
        else:
             print("✅ ChromaDB 연결 확인 성공")

    except ValueError as e:
        # API 키 오류 등 치명적 오류 처리
        print(f"❌ 치명적 오류: RAG 초기화 실패 - {e}")
        rag_retriever = None # 초기화 실패 시 None으로 설정
        # raise
    except Exception as e:
        print(f"❌ 예상치 못한 오류로 RAG 초기화 실패: {e}")
        rag_retriever = None


@app.get("/health", response_model=Dict[str, str])
def health_check() -> Dict[str, str]:
    """API 상태 및 종속성 상태를 확인합니다."""
    status: Dict[str, str] = {"api_status": "ok"}
    
    # RAGRetriever 초기화 성공 여부 확인
    if rag_retriever is None:
        status["rag_status"] = "uninitialized"
        status["detail"] = "RAGRetriever가 초기화되지 않았거나 실패했습니다."
    else:
        status["rag_status"] = "ready"
        
    # ChromaDB 연결 상태 확인 (Optional)
    try:
        if rag_retriever and rag_retriever.vdb_client.health_check():
            status["chroma_status"] = "ok"
        else:
            status["chroma_status"] = "down"
    except Exception:
        status["chroma_status"] = "error"
        
    return status


@app.post("/ask", response_model=ResponseModel)
async def ask_rag(query: QueryModel, request: Request) -> ResponseModel:
    """사용자 질문에 대해 RAG 파이프라인을 실행하여 답변을 제공합니다."""
    
    if rag_retriever is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG 서비스 초기화 실패. 환경 변수(API KEY)를 확인하세요."
        )

    question: str = query.question
    start_time: float = time.time()
    
    try:
        # RAGRetriever를 사용하여 답변 생성
        response: Dict[str, Any] = rag_retriever.answer_query(question)
        
        end_time: float = time.time()
        execution_time_ms: int = int((end_time - start_time) * 1000)

        # 응답 스키마에 맞게 데이터 반환
        return ResponseModel(
            answer=response.get("answer", "답변 생성 실패"),
            source_urls=response.get("source_urls", []),
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        # RAG 처리 중 발생한 예외
        print(f"RAG 처리 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"RAG 파이프라인 실행 중 오류가 발생했습니다: {str(e)}"
        )