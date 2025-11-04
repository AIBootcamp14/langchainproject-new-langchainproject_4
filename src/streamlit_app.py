# src/streamlit_app.py

"""
Streamlit 기반의 RAG 웹 인터페이스
"""

import os
import json
import requests # API 통신을 위해 requests 임포트
from typing import List, Dict, Any, Optional

# 써드파티 라이브러리
import streamlit as st
from dotenv import load_dotenv

# 환경 변수 로드 (로컬 개발 환경용)
load_dotenv()

# --- 설정 및 상수 (PEP 8) ---
# 💡 [핵심 수정]: FastAPI URL을 환경 변수에서 가져오도록 변경
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000") 
API_HEALTH_ENDPOINT: str = f"{FASTAPI_URL}/health"
API_ASK_ENDPOINT: str = f"{FASTAPI_URL}/ask"

# --- 유틸리티 함수 ---

def health_check() -> bool:
    """FastAPI 서버의 헬스 체크 상태를 확인"""
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status() # 200 이외의 상태 코드는 예외 발생
        data = response.json()
        
        # FastAPI의 rag_status와 chroma_status를 모두 확인
        if data.get("rag_status") == "ready" and data.get("chroma_status") == "ok":
             return True
        else:
             st.error(f"FastAPI 서버 준비 중: {data.get('detail', '상세 정보 없음')}")
             return False
             
    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI 서버에 연결할 수 없습니다. URL: {FASTAPI_URL}")
        st.error(f"오류: {e}")
        return False
        
def ask_query(question: str) -> Dict[str, Any]:
    """FastAPI /ask 엔드포인트에 질문을 보내고 결과를 받는다."""
    payload: Dict[str, str] = {"question": question}
    
    try:
        response = requests.post(API_ASK_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API 요청 오류 ({e.response.status_code}): {e.response.json().get('detail', '상세 오류 없음')}")
        return {"answer": "API 요청 처리 중 오류가 발생했습니다.", "source_urls": [], "execution_time_ms": 0}
        
    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI 서버 통신 중 오류 발생: {e}")
        return {"answer": "서버 통신 오류로 답변을 받을 수 없습니다.", "source_urls": [], "execution_time_ms": 0}


# --- Streamlit UI 구성 ---

st.set_page_config(
    page_title="LangChain RAG 챗봇",
    layout="wide"
)

def main_ui():
    """메인 UI를 구성하고 대화 로직을 처리한다."""
    st.title("📚 LangChain 문서 RAG 챗봇")
    st.caption(f"Powered by Solar LLM & ChromaDB via FastAPI ({FASTAPI_URL})")
    
    # 세션 상태 초기화 (대화 기록)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # 1. 헬스 체크
    if not health_check():
        st.warning("FastAPI 백엔드가 준비될 때까지 기다려 주세요.")
        return

    # 2. 이전 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 💡 [핵심 수정]: 실행 시간 정보를 UI에 표시
            if message["role"] == "assistant" and "time" in message:
                st.info(f"⏱️ 응답 시간: {message['time']:.2f}초")


    # 3. 사용자 입력 처리
    if prompt := st.chat_input("LangChain 문서에 대해 질문하세요..."):
        
        # 사용자 질문 표시 및 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 챗봇 답변 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # FastAPI에 질문 전송
                api_response = ask_query(prompt)
                
                answer: str = api_response["answer"]
                source_urls: List[str] = api_response["source_urls"]
                execution_time_ms: int = api_response["execution_time_ms"]
                execution_time_sec: float = execution_time_ms / 1000.0 # 초 단위로 변환
                
                # 답변 출력
                st.markdown(answer)
                
                # 출처 정보 표시
                if source_urls:
                    st.markdown("---")
                    st.markdown("**참조된 출처:**")
                    for url in set(source_urls): # 중복 제거
                        st.markdown(f"- [{url.split('/')[-1]}]({url})")
                
                # 💡 [핵심 수정]: 응답 시간 출력
                st.info(f"⏱️ 응답 시간: {execution_time_sec:.2f}초")

            # 세션 상태에 답변 및 메타데이터 저장
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": source_urls,
                "time": execution_time_sec
            })

if __name__ == "__main__":
    main_ui()