# src/main.py
"""
FastAPI 기반 RAG Chatbot API 서버
- ChromaDB 연결 (VectorDatabaseClient)
- RAG 체인 로드 (retriever)
- /ask 엔드포인트 구현 (질문 및 답변)
"""

import os
from typing import Dict, Any, Optional
import time # os.times() 대신 time.time()을 사용하면 더 범용적

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn # uvicorn 임포트를 여기에 추가 (if __name__ == "__main__" 내부가 아닌 파일 레벨)

# 로컬 모듈 임포트
from .modules.vector_database import get_persisted_vectorstore
from .modules.retriever import get_source_aware_rag_chain

# 환경 변수 로드 (API 키 등)
load_dotenv()

# --- 모델 정의 (Pydantic) ---

class Question(BaseModel):
    """사용자 질문 요청 모델"""
    question: str = Field(..., description="사용자의 질문 내용", min_length=5, max_length=500)
    session_id: Optional[str] = Field("default_session", description="사용자 세션 ID (기억/히스토리 기능용)", max_length=50)


class Answer(BaseModel):
    """LLM 답변 응답 모델"""
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list = Field(..., description="답변에 사용된 출처 문서 목록 (url, title)")
    execution_time_ms: Optional[float] = Field(None, description="실행 시간 (밀리초)")


# --- API 서버 초기화 ---

APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "langchain_docs")

app = FastAPI(
    title="LangChain RAG Chatbot API",
    description="Solar LLM 및 ChromaDB 기반의 LangChain 문서 질의응답 시스템",
    version="1.0.0",
    debug=DEBUG
)

rag_chain = None
vectorstore = None


@app.on_event("startup")
async def startup_event():
    global rag_chain, vectorstore
    
    print("🌟 서버 시작 중: RAG 컴포넌트 초기화 시작")
    
    # 1. Vector Store (ChromaDB) 로드
    try:
        vectorstore = get_persisted_vectorstore(
            host=os.getenv("CHROMA_HOST"), 
            collection_name=COLLECTION_NAME,
        )
        print(f"✅ 벡터 저장소 로드 성공: 컬렉션 '{COLLECTION_NAME}'")
    except ConnectionError as e:
        print(f"❌ ChromaDB 연결 실패: {e}. 'initialize_vector_db.py'를 실행했는지 확인하세요.")
        raise HTTPException(status_code=503, detail="ChromaDB 서버에 연결할 수 없습니다.")
    except Exception as e:
        print(f"❌ 벡터 저장소 초기화 중 예상치 못한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="RAG 시스템 초기화 실패.")

    # 2. RAG 체인 로드
    try:
        rag_chain = get_source_aware_rag_chain(vectorstore=vectorstore)
        print("✅ RAG 체인 로드 성공")
    except Exception as e:
        print(f"❌ RAG 체인 생성 실패 (LLM/임베딩 오류 가능성): {e}")
        raise HTTPException(status_code=500, detail="RAG 체인 로드 실패.")
        
    print("✅ RAG 컴포넌트 초기화 완료")


@app.get("/health", summary="서버 및 DB 상태 확인")
async def health_check():
    if vectorstore is None or rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
        
    # ChromaDB 연결 상태 재확인
    db_status = "OK"
    try:
        # vectorstore.client.heartbeat() 대신 더 안전한 health check
        vectorstore.client.count_collections() 
    except Exception:
        db_status = "DOWN"

    return {
        "status": "OK",
        "service": "RAG Chatbot API",
        "db_status": db_status
    }


@app.post("/ask", response_model=Answer, summary="질문에 답변 생성")
async def ask_question(query: Question):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 아직 로드되지 않았습니다.")
        
    print(f"\n[요청] 세션 ID: {query.session_id}, 질문: {query.question[:50]}...")
    
    try:
        start_time = time.time() # 시간 측정 시작

        result: Dict[str, Any] = rag_chain.invoke({"question": query.question})
        
        end_time = time.time()
        execution_time_ms = round((end_time - start_time) * 1000, 2)
        
        # 출처 문서 정보 추출
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "url": doc.metadata.get("url"),
                "title": doc.metadata.get("title"),
            })

        return Answer(
            answer=result["answer"],
            sources=sources,
            execution_time_ms=execution_time_ms
        )

    except Exception as e:
        print(f"❌ RAG 체인 실행 중 오류 발생: {e}")
        # LLM API 키 오류, 네트워크 오류 등 상세 에러를 숨기지 않고 반환
        raise HTTPException(status_code=500, detail=f"질문 처리 중 서버 오류가 발생했습니다: {e.__class__.__name__}")


# --- 서버 실행 ---
if __name__ == "__main__":
    
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    print(f"🚀 Uvicorn 서버 시작: http://{api_host}:{api_port}")
    
    uvicorn.run(
        "src.main:app", # src 폴더 내의 main.py 파일에서 app 객체를 찾음
        host=api_host, 
        port=api_port, 
        workers=workers, 
        reload=DEBUG,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )