# src/demo.py

import streamlit as st
import requests
import json
from typing import List, Dict, Any

# 💡 FastAPI 서버 엔드포인트 설정 (Docker Compose 환경과 맞춤)
# 로컬에서 개발할 때는 http://localhost:8000/ask 를 사용하면 돼.
API_URL = "http://localhost:8000/ask"
API_HEALTH_CHECK = "http://localhost:8000/"

# --- 1. API 통신 함수 ---
def get_chatbot_response(question: str) -> Dict[str, Any]:
    """
    FastAPI 서버의 /ask 엔드포인트로 질문을 보내고 응답을 받는다.
    """
    try:
        # Pydantic 스키마에 맞게 JSON 데이터 준비
        payload = {"question": question}
        
        # FastAPI 서버로 POST 요청
        response = requests.post(API_URL, json=payload, timeout=300)
        
        # HTTP 응답 코드가 200 (OK)인지 확인
        if response.status_code == 200:
            return response.json()
        
        # 200이 아닐 경우 에러 메시지 반환
        st.error(f"API 호출 실패 (HTTP {response.status_code}): {response.text}")
        return {"answer": "죄송합니다. API 서버에서 오류가 발생했습니다.", "sources": []}

    except requests.exceptions.ConnectionError:
        st.error(f"FastAPI 서버 ({API_URL})에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.")
        return {"answer": "서버 연결 오류.", "sources": []}
    except Exception as e:
        st.error(f"알 수 없는 오류 발생: {e}")
        return {"answer": "예상치 못한 오류가 발생했습니다.", "sources": []}


# --- 2. Streamlit UI 렌더링 함수 ---
st.set_page_config(
    page_title="LangChain RAG 챗봇",
    layout="wide"
)

# 챗봇 제목 및 설명
st.title("📚 LangChain 문서 기반 RAG 챗봇")
st.caption("개발자를 위한 기술 문서(LangChain) 질의응답 서비스 | Powered by Solar API")


# 세션 상태 초기화 (대화 기록 저장)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! LangChain 문서에 대해 궁금한 점을 질문해 주세요."}
    ]


# 서버 상태 체크 (선택 사항: 배포 시 유용)
@st.cache_data(ttl=60)
def check_server_status():
    try:
        requests.get(API_HEALTH_CHECK, timeout=5)
        return True
    except:
        return False

# if check_server_status():
#     st.success("API 서버 상태: 연결됨")
# else:
#     st.warning("API 서버 상태: 연결 불가. Docker Compose를 실행했는지 확인해 주세요.")


# --- 3. 대화 기록 및 입력 처리 ---

# 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자 입력 처리
if prompt := st.chat_input("LangChain 관련 질문을 입력하세요..."):
    
    # 1. 사용자 질문을 기록 및 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 챗봇 응답 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            
            # API 호출
            api_response = get_chatbot_response(prompt)
            
            answer = api_response.get("answer", "응답을 받을 수 없습니다.")
            sources: List[str] = api_response.get("sources", [])
            
            # 답변 출력
            st.markdown(answer)
            
            # 출처 정보 출력 (필수 기능)
            if sources:
                st.markdown("---")
                st.markdown("##### 📌 참조 출처")
                for i, source in enumerate(sources, 1):
                    # 출처 링크를 markdown 형식으로 예쁘게 표시
                    st.markdown(f"{i}. [{source}]({source})")
        
        # 3. 챗봇 응답을 기록 (멀티턴 기능을 위해)
        # 출처를 포함한 최종 마크다운 텍스트를 저장
        full_content = answer
        if sources:
            full_content += "\n\n---\n\n##### 📌 참조 출처\n" + "\n".join(
                [f"{i}. [{s}]({s})" for i, s in enumerate(sources, 1)]
            )
            
        st.session_state.messages.append({"role": "assistant", "content": full_content})