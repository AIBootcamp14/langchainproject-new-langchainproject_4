import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from jinja2 import Template

# ------------------------------------------------------
# 1️⃣ 환경 설정
# ------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "./chroma_langchain_docs"

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------
# 2️⃣ 프롬프트 템플릿 정의
# ------------------------------------------------------
SYSTEM_PROMPT = """
당신은 LangChain 공식 문서를 기반으로 한 AI 챗봇입니다.
사용자의 질문에 대해, 제공된 문서 내용을 바탕으로 정확하고 구체적으로 답변하세요.

규칙:
1. 문서 내용만을 근거로 답합니다.
2. 문서에 없는 내용은 "공식 문서에 해당 내용이 없습니다."라고 답합니다.
3. 관련 문서의 출처를 간단히 표시합니다.
4. 가능한 경우 LangChain 문서 스타일의 코드 예시를 포함합니다.
"""

USER_TEMPLATE = Template("""
질문: {{ question }}

[검색된 문서 요약]
{% for doc in docs %}
출처: {{ doc.metadata.get("source", "unknown") }}
내용 요약:
{{ doc.page_content[:800] }}
---
{% endfor %}

위 문서들을 바탕으로 사용자의 질문에 대한 답변을 작성하세요.
""")

# ------------------------------------------------------
# 3️⃣ Retriever 함수 정의
# ------------------------------------------------------
def get_retriever():
    """기존에 구축된 Chroma DB에서 검색기를 반환"""
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 5})

# ------------------------------------------------------
# 4️⃣ 프롬프트 빌더
# ------------------------------------------------------
def build_prompt(question: str, docs: list):
    """
    검색된 문서 목록(docs)을 템플릿에 삽입해 LLM에 전달할 메시지 포맷 생성
    """
    user_prompt = USER_TEMPLATE.render(question=question, docs=docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    return messages

# ------------------------------------------------------
# 5️⃣ LLM 호출 함수
# ------------------------------------------------------
def call_llm(messages):
    """OpenAI GPT 호출"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------------
# 6️⃣ RAG 통합 파이프라인
# ------------------------------------------------------
def rag_answer(question: str):
    """
    전체 RAG 흐름:
    1. 사용자의 질문을 입력받음
    2. 검색기로 관련 문서 검색
    3. 프롬프트 구성
    4. LLM에 전달하여 답변 생성
    """
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(question)
    messages = build_prompt(question, docs)
    answer = call_llm(messages)
    return answer

# ------------------------------------------------------
# 7️⃣ 실행 예시
# ------------------------------------------------------
if __name__ == "__main__":
    print("🔍 LangChain 문서 기반 RAG 챗봇\n")
    while True:
        query = input("질문: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n🧠 답변:")
        print(rag_answer(query))
        print("-" * 80)
