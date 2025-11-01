import streamlit as st
import requests
import json
import os

# FastAPI 서버의 주소 (Docker 내부에서 접근 시)
# Streamlit 앱이 실행되는 컨테이너에서 접근할 때는 'localhost:8080' 대신
# 컨테이너 이름(rag-application)과 포트(8080)를 사용해야 하지만, 
# Streamlit이 호스트 머신에서 돌아가고 API 호출을 호스트의 8080으로 하므로, 
# 'http://localhost:8080'을 그대로 사용합니다.
FASTAPI_URL = "http://localhost:8080/ask"

def send_question_to_api(question: str) -> dict:
    """
    FastAPI 서버의 /ask 엔드포인트로 질문을 보내고 응답을 받습니다.
    """
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        
        # FastAPI 서버로 POST 요청 전송
        response = requests.post(FASTAPI_URL, headers=headers, data=json.dumps(payload), timeout=300)
        
        # 응답 코드가 200이 아니면 오류 처리
        if response.status_code != 200:
            st.error(f"API 요청 실패: HTTP 상태 코드 {response.status_code}")
            st.json(response.json())
            return {"answer": f"API 오류 발생: 상태 코드 {response.status_code}", "sources": []}
            
        return response.json()
        
    except requests.exceptions.Timeout:
        st.error("API 요청 시간 초과 (300초). 서버 응답이 너무 오래 걸립니다.")
        return {"answer": "처리 시간 초과. 다시 시도해 주세요.", "sources": []}
    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인해 주세요 (포트 8080).")
        return {"answer": "서버 연결 오류. 관리자에게 문의하세요.", "sources": []}
    except Exception as e:
        st.error(f"예기치 않은 오류 발생: {e}")
        return {"answer": f"예기치 않은 오류 발생: {e}", "sources": []}

# --- Streamlit UI 설정 ---

st.set_page_config(page_title="LangChain RAG 챗봇", layout="wide")

# 로고 및 제목
st.markdown("""
    <style>
    .st-emotion-cache-18ni7ap { width: 100% !important; }
    .st-emotion-cache-1avcm0c { background: #f0f2f6; border-radius: 8px; padding: 20px; }
    </style>
    <div style="text-align: center;">
        <h1 style="color: #4A90E2;">🤖 LangChain RAG 챗봇</h1>
        <p style="font-size: 1.1em; color: #555;">LangChain 문서 기반 질의응답 시스템</p>
    </div>
    """, unsafe_allow_html=True)


# 세션 상태 초기화 (대화 기록 저장)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! LangChain 문서에 대해 무엇이든 물어보세요."}
    ]
    
# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("LangChain 관련 질문을 입력하세요..."):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API 호출 및 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            
            # FastAPI에 질문 전송
            response_data = send_question_to_api(prompt)
            
            answer = response_data.get("answer", "답변을 가져오는 데 문제가 발생했습니다.")
            sources = response_data.get("sources", [])
            exec_time = response_data.get("execution_time_ms")
            
            # 답변 출력
            st.markdown(answer)
            
            # 출처 정보 출력
            if sources:
                st.subheader("📚 출처 정보")
                
                # 중복 URL 제거 및 정리
                unique_sources = []
                seen_urls = set()
                
                for source in sources:
                    url = source.get("url")
                    title = source.get("title", url)
                    
                    if url and url not in seen_urls:
                        unique_sources.append(f"- [{title}]({url})")
                        seen_urls.add(url)
                
                # 출처를 리스트로 표시
                st.markdown("\n".join(unique_sources))

        # 답변을 세션 상태에 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})