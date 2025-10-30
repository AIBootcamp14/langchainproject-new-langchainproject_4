from fastapi import FastAPI
from dotenv import load_dotenv
import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.vectorstores import VectorStore
from uvicorn import run

# 프로젝트 모듈 임포트
from src.modules.vector_database import get_persisted_vectorstore
from src.modules.retriever import get_source_aware_rag_chain, RAGChain 

# 💡 Pydantic 모델 정의
class QuestionRequest(BaseModel):
    """API로 들어오는 사용자 질문 요청 스키마"""
    question: str = Field(..., description="사용자가 챗봇에게 던지는 질문")

class AnswerResponse(BaseModel):
    """RAG 챗봇의 답변 및 참조 출처를 포함하는 응답 스키마"""
    answer: str = Field(..., description="LLM이 생성한 최종 답변")
    sources: List[str] = Field(..., description="답변에 사용된 문서의 출처 URL 리스트")


# .env 파일 로드 (SOLAR_API_KEY 등을 환경 변수로 가져옴)
load_dotenv() 

# 전역 변수: 서버 시작 시 초기화될 핵심 객체들
VECTOR_STORE: Optional[VectorStore] = None
RAG_CHAIN: Optional[RAGChain] = None

app = FastAPI(
    title="LangChain RAG Chatbot API",
    description="기술 문서 기반 질의응답을 위한 End-to-End RAG API 서비스"
)

# 💡 FastAPI 서버 시작 시 RAG 코어 초기화
@app.on_event("startup")
def startup_event() -> None:
    """
    FastAPI 서버 시작 시 Vector DB 연결 및 RAG 체인 초기화.
    """
    global VECTOR_STORE, RAG_CHAIN
    
    print("🚀 서버 시작: RAG 컴포넌트 초기화 중...")
    
    # 1. VectorStore 로드 (팀원 5 통합 결과)
    try:
        db_host = os.getenv("CHROMA_HOST")
        VECTOR_STORE = get_persisted_vectorstore(host=db_host)
        print(f"✅ VectorStore 로드 성공. 호스트: {db_host}")
    except ConnectionError as e:
        print(f"❌ DB 연결 실패: {e}")
        VECTOR_STORE = None 
        
    # 2. RAG Chain 초기화 (팀원 2 통합 결과)
    if VECTOR_STORE:
        RAG_CHAIN = get_source_aware_rag_chain(vectorstore=VECTOR_STORE)
        print("✅ RAG Chain 초기화 완료.")
    else:
        RAG_CHAIN = None
        print("⚠️ RAG Chain 초기화 건너뜀. DB 연결 실패.")


@app.get("/")
def health_check():
    """API 서버 상태 확인용 엔드포인트"""
    db_status = "OK" if VECTOR_STORE else "ERROR (DB not connected)"
    return {"status": "ok", "db_status": db_status, "message": "RAG API is running."}


@app.post("/ask", response_model=AnswerResponse)
async def ask_chatbot(request: QuestionRequest):
    """사용자 질문을 받아 RAG 체인을 통해 답변과 출처를 반환"""
    
    if RAG_CHAIN is None:
        error_msg = "RAG 서비스가 준비되지 않았습니다. DB 연결 상태를 확인해주세요."
        return AnswerResponse(answer=error_msg, sources=[])
    
    question = request.question
    
    # RAG 코어 로직 실행
    try:
        response = RAG_CHAIN.invoke(question) 
        
        answer = response["answer"]
        # 출처(Source) 정보 추출 및 중복 제거
        sources = list(set(
            doc.metadata.get("source") for doc in response["docs"] if doc.metadata.get("source")
        ))
    except Exception as e:
        print(f"RAG Chain 실행 오류: {e}")
        answer = f"죄송합니다. RAG 처리 중 예상치 못한 오류가 발생했습니다: {e.__class__.__name__}"
        sources = []

    return AnswerResponse(
        answer=answer,
        sources=sources
    )

if __name__ == "__main__":
    # 개발 환경 실행
    run(app, host="0.0.0.0", port=8000)