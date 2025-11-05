"""
FastAPI 애플리케이션 정의 및 RAG API 엔드포인트
"""

import os
import time
import json 
from typing import Dict, Any, Optional, List

# 써드파티 라이브러리
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request 
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse 

# 프로젝트 모듈
from src.modules.retriever import RAGRetriever 
from src.modules.vector_database import VectorDatabaseClient 

# 환경 변수 미리 로드
load_dotenv()

# --- Pydantic 모델 정의 ---
class QueryModel(BaseModel):
    """사용자 질문을 위한 입력 스키마"""
    question: str = Field(..., description="사용자의 RAG 질문")

class ResponseModel(BaseModel):
    """RAG 답변 및 메타데이터를 위한 출력 스키마 (비스트리밍용)"""
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

# 💡 [새로 추가된 상수]: 재시도 설정
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 3


@app.on_event("startup")
async def startup_event():
    """
    FastAPI 서버 시작 시 RAGRetriever를 초기화하고 종속성을 확인합니다.
    ChromaDB 연결에 성공할 때까지 재시도합니다.
    """
    global rag_retriever
    print("\n--- FastAPI Startup: RAG 파이프라인 초기화 중 (ChromaDB 재시도 포함) ---")
    
    # 💡 [핵심 수정]: ChromaDB 연결을 위한 재시도 로직 추가
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ChromaDB 헬스 체크를 위한 클라이언트를 초기화 (연결 실패 시 여기서 예외 발생 가능)
            vdb_client_check = VectorDatabaseClient(
                collection_name="langchain_docs",
                embedding_model="solar-embedding-1-large"
            )
            
            # ChromaDB 연결 테스트
            if vdb_client_check.health_check():
                print(f"✅ ChromaDB 연결 확인 성공 (시도 {attempt}회)")
                break  # 연결 성공, 루프 탈출
            else:
                # health_check가 False를 반환할 경우 재시도
                raise Exception("ChromaDB health check returned False.")

        except Exception as e:
            # 연결 실패 (ChromaDB가 아직 준비되지 않은 경우)
            print(f"⚠️ ChromaDB 연결 시도 실패 (시도 {attempt}/{MAX_RETRIES}회): {type(e).__name__}. {e}")
            
            if attempt == MAX_RETRIES:
                print("❌ 치명적 오류: ChromaDB 연결 최대 재시도 횟수 초과. RAG 초기화를 건너뜁니다.")
                return # 최종 실패 시 리턴
            
            print(f"⏳ {RETRY_DELAY_SECONDS}초 후 재시도...")
            time.sleep(RETRY_DELAY_SECONDS)
    
    # 재시도 루프를 성공적으로 통과했을 경우에만 RAGRetriever 초기화
    if attempt <= MAX_RETRIES:
        try:
            # RAGRetriever 초기화 (LLM, 임베딩, DB 연결)
            rag_retriever = RAGRetriever()
            print("✅ RAGRetriever 초기화 성공")

        except ValueError as e:
            # API 키 오류 등 치명적 오류 처리
            print(f"❌ 치명적 오류: RAG 초기화 실패 - {e}")
            rag_retriever = None 
        except Exception as e:
            print(f"❌ 예상치 못한 오류로 RAG 초기화 실패: {e}")
            rag_retriever = None


@app.get("/health", response_model=Dict[str, str])
def health_check() -> Dict[str, str]:
    """API 상태 및 종속성 상태를 확인합니다."""
    status: Dict[str, str] = {"api_status": "ok"}
    
    if rag_retriever is None:
        status["rag_status"] = "uninitialized"
        status["detail"] = "RAGRetriever가 초기화되지 않았거나 실패했습니다. ChromaDB/API KEY 확인."
    else:
        status["rag_status"] = "ready"
        
    try:
        # RAGRetriever가 초기화되었을 때만 ChromaDB 상태 확인
        if rag_retriever and rag_retriever.vdb_client.health_check():
            status["chroma_status"] = "ok"
        else:
            status["chroma_status"] = "down"
    except Exception:
        status["chroma_status"] = "error"
        
    return status


@app.post("/ask", response_model=ResponseModel)
async def ask_rag(query: QueryModel, request: Request) -> ResponseModel:
    """사용자 질문에 대해 RAG 파이프라인을 실행하여 답변을 제공합니다. (비스트리밍)"""
    
    if rag_retriever is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG 서비스 초기화 실패. 환경 변수(API KEY)를 확인하세요."
        )

    question: str = query.question
    start_time: float = time.time()
    
    try:
        response: Dict[str, Any] = rag_retriever.answer_query(question)
        
        end_time: float = time.time()
        execution_time_ms: int = int((end_time - start_time) * 1000)

        return ResponseModel(
            answer=response.get("answer", "답변 생성 실패"),
            source_urls=response.get("source_urls", []),
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        print(f"RAG 처리 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"RAG 파이프라인 실행 중 오류가 발생했습니다: {str(e)}"
        )


# 💡 [핵심 추가]: 스트리밍 엔드포인트
@app.post("/ask/stream")
async def ask_rag_stream(query: QueryModel) -> StreamingResponse:
    """사용자 질문에 대해 RAG 파이프라인을 실행하고 답변을 스트리밍으로 제공합니다."""
    
    if rag_retriever is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG 서비스 초기화 실패. 환경 변수(API KEY)를 확인하세요."
        )

    question: str = query.question
    
    async def generate_stream():
        start_time: float = time.time()
        METADATA_DELIMITER = "\n<END_OF_STREAM_METADATA>" # 스트림 종료 메타데이터 구분자
        
        try:
            # 1. 검색된 문서 미리 가져오기 (출처 URL 추출 및 컨텍스트 구성을 위해)
            # RAGRetriever.retriever는 VectorStoreRetriever이며, ainvoke를 지원함
            retrieved_docs: List[Document] = await rag_retriever.retriever.ainvoke(question)

            # 2. RAG 체인을 비동기 스트림(astream)으로 호출
            stream = rag_retriever.rag_chain.astream(question)
            
            # 3. 답변 스트림을 클라이언트에 전송
            async for chunk in stream:
                # 각 청크(문자열)를 인코딩하여 전송
                yield chunk.encode("utf-8")
                
            # 4. 스트림 종료 후 메타데이터를 포함한 최종 데이터 전송
            end_time: float = time.time()
            execution_time_ms: int = int((end_time - start_time) * 1000)
            
            # 출처 URL 추출 (중복 제거)
            source_urls = list(
                set(
                    doc.metadata["url"] 
                    for doc in retrieved_docs 
                    if "url" in doc.metadata
                )
            )
            
            # 메타데이터를 JSON 형태로 전송 (특수 구분자로 본문과 구분)
            metadata = {
                "source_urls": source_urls,
                "execution_time_ms": execution_time_ms
            }
            yield f"{METADATA_DELIMITER}{json.dumps(metadata)}".encode("utf-8")

        except Exception as e:
            error_message = f"RAG 스트림 처리 중 오류 발생: {str(e)}"
            # 에러 메시지를 메타데이터 형식으로 전달하여 클라이언트가 처리하도록 유도
            yield f"{METADATA_DELIMITER}{json.dumps({'error': error_message})}".encode("utf-8")
            
    # 스트리밍 응답 반환
    return StreamingResponse(generate_stream(), media_type="text/plain")