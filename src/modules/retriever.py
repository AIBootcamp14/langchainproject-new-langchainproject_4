from typing import List, Dict, Any

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from .llm import get_solar_llm
from .prompts import RAG_PROMPT
from langchain_core.language_models import BaseChatModel

# 타입 힌트를 위한 별칭
RAGChain = Runnable[str, Dict[str, Any]]
ContextAwareChain = Runnable[str, str]

def format_docs(docs: List[Document]) -> str:
    """
    검색된 Document 객체 리스트를 LLM의 Context 입력에 적합한 하나의 문자열로 포맷팅
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_source_aware_rag_chain(vectorstore: VectorStore) -> RAGChain:
    """
    LCEL을 사용하여 최종 답변과 검색된 출처 정보를 함께 반환하는 RAG 체인을 구성
    
    Args:
        vectorstore: 데이터가 적재된 벡터 저장소 객체

    Returns:
        RAGChain: 사용자 질문(str)을 받아 {'answer': str, 'docs': List[Document]}를 반환하는 체인
    """
    llm: BaseChatModel = get_solar_llm()
    # 팀원 5와 협의하여 k 값(검색 문서 개수)을 설정하기. 일단 3개로 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 1. 검색 단계 (Retrieval): 질문을 받아 Document 리스트를 검색하고, Context로 포맷팅해
    # RunnablePassthrough.assign을 사용해 최종 출력에 'docs'와 'context'를 포함시킴
    setup_and_retrieval = RunnablePassthrough.assign(
        docs=retriever, # 검색된 Document 리스트
    ).assign(
        context=lambda x: format_docs(x["docs"]), # 검색된 문서 내용을 Context 문자열로 포맷팅
        question=RunnablePassthrough(), # 질문 원본을 그대로 전달
    )

    # 2. 생성 단계 (Generation): 검색 결과와 질문을 프롬프트에 넣어 LLM에게 답변을 요청
    answer_generation = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # 3. 최종 체인: 검색 결과와 답변 결과를 합쳐서 반환
    # setup_and_retrieval의 출력에 'answer' 키를 추가하는 방식으로 최종 결과를 구성
    final_chain: RAGChain = setup_and_retrieval.assign(
        answer=answer_generation,
    )
    
    return final_chain