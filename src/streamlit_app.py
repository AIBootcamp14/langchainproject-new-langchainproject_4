"""
Streamlit 기반의 RAG 웹 인터페이스
"""

import os
import json
import requests
import sys
from typing import List, Dict, Any, Optional

# 써드파티 라이브러리
import streamlit as st
from dotenv import load_dotenv

# 환경 변수 로드 (로컬 개발 환경용)
load_dotenv()

# --- 설정 및 상수 (PEP 8) ---
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000")
API_HEALTH_ENDPOINT: str = f"{FASTAPI_URL}/health"
API_ASK_STREAM_ENDPOINT: str = f"{FASTAPI_URL}/ask/stream"

# 상수 정의 (PEP 8: 대문자 사용)
METADATA_DELIMITER: str = "\n<END_OF_STREAM_METADATA>"

# --- 유틸리티 함수 ---

def health_check() -> bool:
    """FastAPI 서버의 헬스 체크 상태를 확인"""
    try:
        response: requests.Response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

        # OpenAPI 서버가 준비되었는지 확인
        if data.get("rag_status") == "ready" and data.get("chroma_status") == "ok":
            return True
        else:
            st.toast(f"FastAPI 서버 준비 중: {data.get('detail', '상세 정보 없음')}", icon="⏳")
            return False

    except requests.exceptions.RequestException as e:
        # st.error(f"FastAPI 서버에 연결할 수 없습니다. URL: {FASTAPI_URL}")
        st.toast("서버 연결 오류. FastAPI 서버가 켜져 있는지 확인하세요.", icon="❌")
        return False

def ask_query_stream(question: str) -> Any:
    """
    FastAPI /ask/stream 엔드포인트에 질문을 보내고 결과를 스트리밍으로 받는다.
    
    Args:
        question: 사용자 질문 문자열.
    
    Yields:
        응답 청크 문자열.
        
    Returns:
        메타데이터 딕셔너리 또는 에러 딕셔너리.
    """
    payload: Dict[str, str] = {"question": question}

    try:
        # 스트리밍 요청
        response: requests.Response = requests.post(
            API_ASK_STREAM_ENDPOINT, 
            json=payload, 
            stream=True, 
            timeout=60
        )
        response.raise_for_status()

        full_answer: str = ""
        
        # Streamlit 메시지 플레이스홀더를 사용한 스트리밍 출력
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue

            # 메타데이터 구분자가 포함된 경우
            if METADATA_DELIMITER in chunk:
                answer_chunk, metadata_json_str = chunk.split(METADATA_DELIMITER, 1)
                full_answer += answer_chunk
                yield answer_chunk # 답변의 마지막 청크

                try:
                    # 메타데이터 파싱 후 반환
                    metadata: Dict[str, Any] = json.loads(metadata_json_str)
                    return metadata
                except json.JSONDecodeError:
                    st.toast("메타데이터 파싱 오류 발생.", icon="⚠️")
                    return {"error": "Metadata parsing failed."}
            else:
                # 일반 답변 청크
                full_answer += chunk
                yield chunk

        # 메타데이터 없이 스트림이 끝난 경우 (예외 처리)
        return {"answer": full_answer, "source_urls": [], "execution_time_ms": 0}

    except requests.exceptions.HTTPError as e:
        # HTTP 에러 발생 시 처리
        error_detail: str = e.response.json().get('detail', '상세 오류 없음')
        return {"error": f"API 요청 오류 ({e.response.status_code}): {error_detail}"}
    except requests.exceptions.RequestException as e:
        # 기타 통신 오류 처리
        return {"error": f"FastAPI 서버 통신 중 오류 발생: {e}"}


# --- 세션 초기화 함수 ---

def initialize_session_state() -> None:
    """세션 상태를 초기화합니다. (messages만 없으면 초기화)"""
    # 💡 [핵심 수정]: 세션이 비어있을 때만 초기화하여 불필요한 리셋 방지
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 디버깅 출력은 이제 필요 없어, 불필요한 메시지 생성을 막기 위해 제거
        st.toast("새로운 채팅 세션 시작.", icon="👋")


# --- Streamlit UI 구성 ---

st.set_page_config(
    page_title="LangChain RAG 챗봇",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

def main_ui() -> None:
    """메인 UI를 구성하고 대화 로직을 처리한다."""
    # 💡 CSS 스타일링은 HTML 마크다운 대신 st.markdown으로 유지
    st.markdown(
        """
        <style>
        /* 전체 페이지 배경 및 폰트 */
        .stApp {
            background-color: #f4f7f9; /* 옅은 회색 배경 */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* 나머지 CSS는 동일하게 유지 */
        .stChatMessage {
            border-radius: 12px;
            padding: 10px 15px;
            margin-bottom: 10px;
        }
        .stChatMessage[data-testid="stChatMessage"][data-element-type="chat-message"][data-is-user="true"] {
            background-color: #e6f7ff;
            border-left: 5px solid #007bff;
        }
        .stChatMessage[data-testid="stChatMessage"][data-element-type="chat-message"][data-is-user="false"] {
            background-color: #ffffff;
            border-right: 5px solid #007bff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stChatInput {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            background: #f4f7f9;
            z-index: 1000;
            border-top: 1px solid #ddd;
        }
        .chat-history-container {
            height: 75vh;
            overflow-y: auto;
            padding-bottom: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 LangChain 문서 RAG 챗봇")
    st.caption(f"Powered by Solar LLM & ChromaDB via FastAPI ({FASTAPI_URL})")

    # 0. 세션 상태 초기화
    initialize_session_state()

    # 1. 헬스 체크 및 서버 상태 표시
    if not health_check():
        st.stop()

    # 2. 대화 기록 표시 컨테이너
    chat_history_container = st.container(height=500, border=False)

    with chat_history_container:
        # 대화 기록을 화면에 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 어시스턴트 메시지에만 소스 및 시간 정보 표시
                if message["role"] == "assistant":
                    if "time" in message:
                        st.info(f"⏱️ 응답 시간: {message['time']:.2f}초")

                    if "sources" in message and message["sources"]:
                        with st.expander("참조된 출처 보기"):
                            # 중복 URL 제거 및 표시
                            for url in sorted(list(set(message["sources"]))):
                                file_name: str = url.split('/')[-1] if url.split('/')[-1] else url
                                st.markdown(f"- [{file_name}]({url})")


    # 3. 사용자 입력 처리
    if prompt := st.chat_input("LangChain 문서에 대해 질문하세요..."):

        # 1차: 질문 저장
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 답변 생성을 위해 즉시 rerun
        st.rerun()


    # 4. 답변 생성 및 저장 (RERUN 2: 답변 생성)
    # 마지막 메시지가 사용자 메시지이고, 답변이 아직 생성되지 않았을 때만 실행
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user"):

        current_prompt: str = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response: str = ""
            final_metadata: Dict[str, Any] = {}
            
            # 답변 스트리밍 시작
            stream_generator = ask_query_stream(current_prompt)

            try:
                for chunk in stream_generator:
                    if isinstance(chunk, str):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # 스트리밍 완료 후 메타데이터 반환 받기
                if isinstance(stream_generator, dict):
                    final_metadata = stream_generator
                else:
                    # Generator가 정상 종료되어 메타데이터를 반환
                    final_metadata = next(stream_generator, {}) # Generator의 마지막 반환 값을 가져옴
                    
            except Exception as e:
                # 💡 API 통신 오류 발생 시 처리
                print(f"--- ERROR: Streaming failed. Exception: {e}", file=sys.stderr)
                st.error(f"스트리밍 중 오류 발생: {e}")

            # 최종 응답 출력
            message_placeholder.markdown(full_response)

            # 답변 저장
            if "error" not in final_metadata:
                source_urls: List[str] = final_metadata.get("source_urls", [])
                execution_time_ms: int = final_metadata.get("execution_time_ms", 0)
                execution_time_sec: float = execution_time_ms / 1000.0

                st.info(f"⏱️ 응답 시간: {execution_time_sec:.2f}초 (스트리밍 포함)")
                st.toast("답변 생성 완료!", icon="✅")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": source_urls,
                    "time": execution_time_sec
                })
            else:
                # 오류 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"죄송합니다. API 통신 오류로 답변을 생성할 수 없습니다. ({final_metadata.get('error', '알 수 없는 오류')})",
                    "sources": [],
                    "time": 0
                })
                st.error(f"API 통신 오류로 답변 생성 실패: {final_metadata.get('error', '알 수 없는 오류')}")

        # 💡 [핵심 수정]: st.rerun() 제거!
        # 답변이 저장된 후 다시 렌더링할 필요 없음. Streamlit이 알아서 다음 입력을 기다림.


if __name__ == "__main__":
    main_ui()