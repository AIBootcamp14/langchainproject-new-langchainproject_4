# src/main.py

from fastapi import FastAPI
from dotenv import load_dotenv
from .schemas.rag_schema import QuestionRequest, AnswerResponse
from typing import List

# .env 파일 로드 (SOLAR_API_KEY 등을 환경 변수로 가져옴)
load_dotenv() 

# 💡 참고: LLM 및 RAG 모듈은 나중에 통합할 거야
# from .modules.retriever import get_rag_response 

app = FastAPI(
    title="LangChain RAG Chatbot API",
    description="기술 문서 기반 질의응답을 위한 End-to-End RAG API 서비스"
)

@app.get("/")
def health_check():
    """
    API 서버 상태 확인용 엔드포인트
    """
    return {"status": "ok", "message": "RAG API is running."}


@app.post("/ask", response_model=AnswerResponse)
async def ask_chatbot(request: QuestionRequest):
    """
    사용자 질문을 받아 RAG 체인을 통해 답변과 출처를 반환
    """
    question = request.question
    
    # 📌 TODO: 팀원 2의 RAG 코어 로직 통합 위치
    # answer, sources = get_rag_response(question)
    
    # 임시 응답 (팀원 2 작업이 완료되기 전까지 사용)
    # 팀원 2, 3이 API 테스트를 할 수 있도록 명세에 맞는 더미 응답을 반환
    dummy_answer = f"FastAPI 서버 작동 확인: 질문 '{question}'에 대한 답변 준비 중입니다."
    dummy_sources: List[str] = [
        "https://docs.langchain.com/oss/python/integrations/splitter-example1", 
        "https://docs.langchain.com/oss/python/integrations/example-source2"
    ]

    return AnswerResponse(
        answer=dummy_answer,
        sources=dummy_sources
    )

# 💡 참고: uvicorn 명령어로 실행됨: uvicorn main:app --host 0.0.0.0 --port 8000