# src/modules/retriever.py

"""
검색 증강 생성(RAG) 파이프라인 구현 모듈
- LCEL을 사용하여 RAG 체인을 구성
- LLM 호출, 검색, 응답 생성 로직 포함
"""

import os
from typing import List, Dict, Any, Final

# 써드파티 라이브러리
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough # Runnable 타입 추가
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

# 프로젝트 모듈
from src.modules.llm import get_solar_llm # Solar LLM 임포트
from src.modules.vector_database import VectorDatabaseClient

# --- 설정 및 상수 ---
COLLECTION_NAME: Final[str] = "langchain_docs"
EMBEDDING_MODEL_NAME: Final[str] = "solar-embedding-1-large"
RETRIEVAL_K: Final[int] = 5 # 검색할 문서 개수


# RAG 답변 생성에 사용할 프롬프트 템플릿 정의 (PEP 8 준수)
RAG_PROMPT_TEMPLATE: Final[str] = """
당신은 LangChain 기술 문서 전문가입니다. 사용자의 질문에 대해 주어진 'context'만을 사용하여 상세하고 정확하게 답변해야 합니다.
만약 주어진 context 내에서 답변을 찾을 수 없다면, '정보를 찾을 수 없습니다.'라고 답변하십시오. 답변에 출처 정보는 절대 포함하지 마십시오.

---
Question: {question}

Context: 
{context}
---
"""


# Document 객체 리스트를 하나의 문자열로 포맷팅하는 헬퍼 함수
def _format_docs(docs: List[Document]) -> str:
    """Retriever에서 반환된 Document 리스트를 LLM에 전달할 컨텍스트 문자열로 포맷한다."""
    # 각 문서의 내용을 합쳐서 반환
    return "\n\n".join([doc.page_content for doc in docs])


class RAGRetriever:
    """
    RAG 파이프라인을 초기화하고 사용자 질문에 대한 답변을 생성하는 클래스.
    """

    def __init__(self) -> None:
        """필요한 구성 요소(LLM, DB 클라이언트, 체인)를 초기화한다."""
        # LLM 초기화 (RAG는 창의성보다 정확도가 중요하므로 온도는 낮게 설정)
        self.llm = get_solar_llm(temperature=0.05) 
        
        # VectorDB 클라이언트 초기화
        self.vdb_client: VectorDatabaseClient = VectorDatabaseClient(
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        
        # Retriever 초기화
        # 여기서 DB 연결이 실제로 발생하며, 실패 시 에러가 발생해야 함 (FastAPI startup에서 처리)
        self.retriever: VectorStoreRetriever = self.vdb_client.get_retriever(k=RETRIEVAL_K)
        
        # LCEL RAG 체인 초기화
        self.rag_chain: Runnable = self._create_rag_chain()

    def _create_rag_chain(self) -> Runnable:
        """
        LCEL (LangChain Expression Language)을 사용하여 RAG 체인을 구성한다.
        """
        # 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # RAG 파이프라인 (LCEL) 구성
        rag_chain = (
            # 1. 입력 (question)을 받아서
            {
                # 2. 'context' 키에는 retriever를 통해 문서 검색 후 포맷팅한 결과물을 넣고
                "context": self.retriever | _format_docs, 
                # 3. 'question' 키에는 질문(입력)을 그대로 통과시킨다.
                "question": RunnablePassthrough() 
            }
            # 4. 프롬프트를 구성하고
            | prompt
            # 5. LLM을 호출하여 답변을 생성하고
            | self.llm
            # 6. 문자열로 파싱한다.
            | StrOutputParser() 
        )
        return rag_chain

    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        사용자 질문에 답변하고 검색된 출처 URL을 반환한다.
        
        Args:
            question: 사용자의 질문 문자열.
            
        Returns:
            답변 및 출처 URL을 포함하는 딕셔너리.
        """
        
        # 1. 검색된 문서 미리 가져오기 (출처 URL 추출을 위해)
        # 💡 [필수]: LLM 체인이 아닌, retriever에서 검색된 결과물을 미리 가져와야 메타데이터를 얻을 수 있음.
        # RAG 체인 실행 시 context를 위해 retriever가 한 번 더 실행될 수 있지만, 
        # 메타데이터를 얻기 위해서는 별도의 retriever.invoke(question)이 필요하다.
        retrieved_docs: List[Document] = self.retriever.invoke(question)

        # 2. RAG 체인 실행 (답변 생성)
        answer: str = self.rag_chain.invoke(question)

        # 3. 출처 URL 추출 (중복 제거)
        source_urls: List[str] = list(
            set(
                doc.metadata["url"] 
                for doc in retrieved_docs 
                if "url" in doc.metadata
            )
        )

        # 4. 결과 반환 (main.py에서 실행 시간 측정)
        return {
            "answer": answer,
            "source_urls": source_urls
        }


if __name__ == "__main__":
    # 테스트 코드는 VectorDatabaseClient와 LLM이 작동할 때만 의미가 있으므로 간단히 작성
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 50)
    print("RAGRetriever 모듈 테스트 시작")
    print("=" * 50)
    
    try:
        # DB가 실행 중이고 데이터가 적재된 후에만 테스트 가능
        rag_retriever = RAGRetriever()
        
        # 간단한 테스트 질문
        test_question = "LCEL을 사용하는 주요 이점은 무엇인가요?"
        
        print(f"테스트 질문: {test_question}")
        
        # answer_query 호출
        response = rag_retriever.answer_query(test_question)
        
        print("\n=== RAG 응답 결과 ===")
        print(f"답변: {response['answer']}")
        print(f"출처 URL: {response['source_urls']}")
        
    except Exception as e:
        print(f"❌ RAGRetriever 테스트 중 오류 발생: {e.__class__.__name__} - {e}")
        print("ChromaDB 서버와 LLM API 키를 확인해 주세요.")