from typing import List, Dict, Any, Optional

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from .llm import get_solar_llm
from .prompts import RAG_PROMPT

# PEP 8: 모듈 수준 상수는 대문자로
DEFAULT_K = 3 # 검색 문서 개수 기본값

# 타입 힌트를 위한 별칭 (PEP 484 준수)
RAGChain = Runnable[Dict[str, Any], Dict[str, Any]]


def format_docs(docs: List[Document]) -> str:
    """
    검색된 Document 객체 리스트를 LLM의 Context 입력에 적합한 하나의 문자열로 포맷팅
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_source_aware_rag_chain(vectorstore: VectorStore, k: int = DEFAULT_K) -> RAGChain:
    """
    LCEL을 사용하여 최종 답변과 검색된 출처 정보를 함께 반환하는 RAG 체인을 구성합니다.
    
    Args:
        vectorstore: 데이터가 적재된 벡터 저장소 객체
        k: 검색할 문서의 개수

    Returns:
        RAGChain: 사용자 질문(str)을 받아 {'answer': str, 'source_documents': List[Document]}를 반환하는 체인
    """
    llm: BaseChatModel = get_solar_llm()
    
    # 💡 [핵심 수정]: vectorstore.as_retriever() 사용 대신, 
    #    Langchain-upstage 라이브러리의 잠재적 버그를 우회하기 위해 
    #    similarity_search를 RunnableLambda를 사용하여 직접 호출합니다.
    retriever = RunnableLambda(
        # x는 {'question': str, 'session_id': str} 형태의 딕셔너리입니다.
        lambda x: vectorstore.similarity_search(x["question"], k=k) 
    ).with_config(run_name="CustomRetrieval")

    # 1. 검색 단계 (Retrieval): 질문을 받아 Document 리스트를 검색하고, Context로 포맷팅
    # RunnablePassthrough.assign을 사용해 최종 출력에 'docs'와 'context'를 포함
    setup_and_retrieval = RunnablePassthrough.assign(
        docs=retriever, # 검색된 Document 리스트
    ).assign(
        context=lambda x: format_docs(x["docs"]), # 검색된 문서 내용을 Context 문자열로 포맷팅
        question=lambda x: x["question"], # 질문 키를 유지
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
    ).assign(
        # 'docs' 키를 'source_documents'로 이름 변경하여 main.py의 응답 스키마와 일치시킴
        source_documents=lambda x: x["docs"],
    ).with_config(
        output_keys=["answer", "source_documents"] # 불필요한 키 제거
    )
    
    return final_chain