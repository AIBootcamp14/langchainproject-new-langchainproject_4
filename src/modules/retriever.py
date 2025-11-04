# src/modules/retriever.py (전면 수정)

from typing import List, Dict, Any

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain의 대화형 체인 생성 헬퍼 임포트 (질문 재구성에만 사용)
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# llm 모듈 및 프롬프트 임포트
from .llm import get_solar_llm, get_solar_sql_llm
from .prompts import RAG_PROMPT, CONDENSE_QUESTION_PROMPT, TRANSLATE_PROMPT

# 타입 힌트
ConversationalRAGChain = Runnable[Dict[str, Any], Dict[str, Any]]

def format_docs(docs: List[Document]) -> str:
    """ (이 함수는 변경 없음) """
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_rag_chain(vectorstore: VectorStore) -> ConversationalRAGChain:
    """
    [업그레이드된 대화형 RAG 체인]
    한국어 질문 -> 한국어 독립 질문 -> 영어 번역 -> 영문서 검색 -> 한국어 답변
    """
    llm: BaseChatModel = get_solar_llm()
    # 번역은 정확해야 하므로 temperature=0.0 사용 (SQL LLM 재활용)
    translation_llm: BaseChatModel = get_solar_sql_llm() 
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- 1. 독립적인 질문 생성 체인 (한국어) ---
    # (입력: question, chat_history -> 출력: standalone_korean_question)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )

    # --- 2. 질문 번역 체인 (한국어 -> 영어) ---
    # (입력: question -> 출력: standalone_english_question)
    translation_chain = (
        TRANSLATE_PROMPT
        | translation_llm
        | StrOutputParser()
    )

    # --- 3. 답변 생성 체인 (RAG의 핵심) ---
    # (입력: context, question -> 출력: answer)
    # RAG_PROMPT는 한국어로 되어 있으므로, 영어 Context와 한국어 Question을 받아
    # 한국어 Answer를 생성합니다.
    answer_chain = create_stuff_documents_chain(
        llm,
        RAG_PROMPT 
    )

    # --- 4. LCEL을 사용하여 위 3개 체인 + 검색(Retriever)을 통합 ---

    # 4.1. 원본 입력을 받아 'standalone_korean_question' 생성
    chain_with_standalone_ko = RunnablePassthrough.assign(
        standalone_korean_question=condense_question_chain
    )

    # 4.2. (1)의 결과(standalone_korean_question)를 'translation_chain'에 전달
    chain_with_standalone_en = chain_with_standalone_ko.assign(
        standalone_english_question=RunnableLambda(
            lambda x: {"question": x["standalone_korean_question"]}
        ) | translation_chain
    )

    # 4.3. (2)의 결과(standalone_english_question)를 'retriever'에 전달하여 'context' 생성
    chain_with_context = chain_with_standalone_en.assign(
        context=RunnableLambda(
            lambda x: x["standalone_english_question"]
        ) | retriever
    )
    
    # 4.4. (3)의 결과('context')와 원본 'question'을 'answer_chain'에 전달
    # 💡 'question' 키는 RunnablePassthrough가 원본 입력을 그대로 전달해줍니다.
    chain_with_answer = chain_with_context.assign(
        answer=answer_chain
    )

    # 4.5. 최종 출력 포맷팅
    final_chain: ConversationalRAGChain = chain_with_answer.assign(
        source_documents=lambda x: x["context"],
    ).with_config(
        output_keys=["answer", "source_documents"] # main.py와 호환
    )
    
    return final_chain