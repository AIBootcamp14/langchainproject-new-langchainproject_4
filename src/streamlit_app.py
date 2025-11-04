# src/streamlit_app.py

"""
Streamlit 기반의 RAG 웹 인터페이스
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

# 써드파티 라이브러리
import streamlit as st
from dotenv import load_dotenv

# 환경 변수 로드 (로컬 개발 환경용)
load_dotenv()

# --- 설정 및 상수 (PEP 8) ---
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000") 
API_HEALTH_ENDPOINT: str = f"{FASTAPI_URL}/health"
API_ASK_STREAM_ENDPOINT: str = f"{FASTAPI_URL}/ask/stream" # 💡 [핵심 수정]: 스트리밍 엔드포인트 사용


# --- 유틸리티 함수 ---

def health_check() -> bool:
    """FastAPI 서버의 헬스 체크 상태를 확인"""
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("rag_status") == "ready" and data.get("chroma_status") == "ok":
             return True
        else:
             st.warning(f"FastAPI 서버 준비 중: {data.get('detail', '상세 정보 없음')}")
             return False
             
    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI 서버에 연결할 수 없습니다. URL: {FASTAPI_URL}")
        st.error(f"오류: {e}")
        return False
        
def ask_query_stream(question: str) -> Dict[str, Any]:
    """
    FastAPI /ask/stream 엔드포인트에 질문을 보내고 결과를 스트리밍으로 받는다.
    
    Yields: 답변 청크 (str)
    Returns: 최종 메타데이터 딕셔너리
    """
    payload: Dict[str, str] = {"question": question}
    
    try:
        # stream=True로 설정하여 스트리밍 연결
        response = requests.post(API_ASK_STREAM_ENDPOINT, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        full_answer = ""
        metadata: Dict[str, Any] = {}
        
        # FastAPI에서 정의한 특수 구분자
        METADATA_DELIMITER = "\n<END_OF_STREAM_METADATA>"

        # 스트림을 청크 단위로 읽음
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            
            # 메타데이터 구분자가 있는지 확인
            if METADATA_DELIMITER in chunk:
                # 본문과 메타데이터 분리
                answer_chunk, metadata_json_str = chunk.split(METADATA_DELIMITER, 1)
                full_answer += answer_chunk
                yield answer_chunk # 마지막 답변 청크 전달
                
                # 메타데이터 파싱
                try:
                    metadata = json.loads(metadata_json_str)
                    metadata["answer"] = full_answer # 최종 답변을 메타데이터에 포함
                except json.JSONDecodeError:
                    st.error("메타데이터 파싱 오류 발생. 서버 응답 확인 필요.")
                    metadata = {"error": "Metadata parsing failed."}
                    
                # 메타데이터를 받았으므로 최종 결과 반환
                return metadata
            else:
                full_answer += chunk
                yield chunk # Streamlit에게 청크를 반환하여 UI에 업데이트되도록 함
        
        # 스트림이 정상적으로 닫혔으나 메타데이터가 없는 경우
        return {"answer": full_answer, "source_urls": [], "execution_time_ms": 0}
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API 요청 오류 ({e.response.status_code}): {e.response.json().get('detail', '상세 오류 없음')}")
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
            # 💡 [핵심 수정]: Streamlit의 empty 컨테이너를 사용하여 실시간 업데이트
            message_placeholder = st.empty()
            full_response = ""
            
            # 1. 스트림 요청 및 실시간 답변 업데이트
            stream_generator = ask_query_stream(prompt)
            
            for chunk in stream_generator:
                if isinstance(chunk, str):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌") # 커서 효과
                
            # 2. 최종 메타데이터 처리 및 UI 업데이트
            # stream_generator가 최종적으로 반환하는 메타데이터를 받는다.
            # Generator의 return 값은 StopIteration 예외의 value로 전달되지만, 
            # 여기서는 ask_query_stream 함수의 반환 값을 명시적으로 처리할 수 없으므로,
            # stream_generator 실행 후 full_response와 별도로 저장된 메타데이터를 사용해야 한다.
            
            # [수정 필요]: stream_generator가 끝난 후 최종 메타데이터를 가져오는 명시적 방법이 필요함.
            # ask_query_stream 함수를 yield로 만들고, 마지막에 return 대신 예외를 활용하거나,
            # 아니면 main_ui에서 generator를 실행하고 마지막 return 값을 명시적으로 받도록 코드를 수정해야 함.
            
            # [임시 수정]: stream_generator가 끝난 후, generator 객체가 반환한 딕셔너리를 직접 받는다.
            try:
                # generator의 최종 딕셔너리를 받음
                final_metadata = next(stream_generator) 
            except StopIteration as e:
                # StopIteration의 value에 return 값이 담겨 있음
                final_metadata = e.value if e.value is not None else {}
            except TypeError:
                 # ask_query_stream이 예외로 끝났을 때 빈 딕셔너리로 처리
                 final_metadata = {} 
            
            
            # 3. 최종 답변 및 커서 제거
            message_placeholder.markdown(full_response)
            
            if final_metadata and not final_metadata.get("error"):
                source_urls: List[str] = final_metadata.get("source_urls", [])
                execution_time_ms: int = final_metadata.get("execution_time_ms", 0)
                execution_time_sec: float = execution_time_ms / 1000.0
                
                # 출처 정보 표시
                if source_urls:
                    st.markdown("---")
                    st.markdown("**참조된 출처:**")
                    for url in set(source_urls):
                        st.markdown(f"- [{url.split('/')[-1]}]({url})")
                
                # 응답 시간 출력
                st.info(f"⏱️ 응답 시간: {execution_time_sec:.2f}초 (스트리밍 포함)")

                # 세션 상태에 답변 및 메타데이터 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": source_urls,
                    "time": execution_time_sec
                })
            elif final_metadata.get("error"):
                 st.error(f"스트리밍 오류: {final_metadata['error']}")
            else:
                 # 오류가 발생했거나 메타데이터를 받지 못한 경우 (ask_query_stream에서 이미 에러를 표시했을 수 있음)
                 pass 


if __name__ == "__main__":
    main_ui()