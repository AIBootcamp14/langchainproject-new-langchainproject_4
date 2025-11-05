"""
벡터 데이터베이스(ChromaDB) 연결 및 관리 클래스
(환경 변수를 모듈 로드 시점이 아닌, 인스턴스 생성 시점에 동적으로 로드하도록 수정)
"""

from typing import Any, Final
import os
from typing import List

# 써드파티 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from chromadb import HttpClient 

# ❌ [제거]: 모듈 수준의 상수를 제거하고, 클래스 내부에서 동적으로 읽도록 함
# CHROMA_HOST: Final[str] = os.getenv("CHROMA_HOST", "localhost") 
# CHROMA_PORT: Final[int] = int(os.getenv("CHROMA_PORT", "8000")) 


class VectorDatabaseClient:
    """ChromaDB 연결, 초기화, 컬렉션 관리를 담당하는 클라이언트."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        """
        Args:
            collection_name: 사용할 ChromaDB 컬렉션 이름.
            embedding_model: 사용할 임베딩 모델 이름 (Solar Embedding).
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # 💡 [핵심 수정]: 인스턴스가 생성되는 시점(initialize_db 호출 후)에 환경 변수를 동적으로 읽음
        # initialize_db에서 설정한 '8001' 포트가 여기서 정확하게 반영됨
        self.chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port: int = int(os.getenv("CHROMA_PORT", "8000")) 
        self.chroma_url: str = f"http://{self.chroma_host}:{self.chroma_port}"
        
        # get_embeddings 함수를 사용하여 Embeddings 인스턴스 초기화
        from src.modules.llm import get_embeddings
        self.embeddings: Embeddings = get_embeddings(model=embedding_model)

    def health_check(self) -> bool:
        """
        ChromaDB 서버 연결 상태를 확인한다.
        """
        try:
            # 구버전 호환을 위해 tenant, database 인수 제거
            client = HttpClient(
                host=self.chroma_host, # 인스턴스 변수 사용
                port=self.chroma_port, # 인스턴스 변수 사용
            )
            client.heartbeat() # 하트비트 호출로 연결 확인
            return True
        except Exception as e: 
            # ❌ [디버그 코드 유지]: 연결 실패 시 오류 출력
            print(f"DEBUG_CHROMA_ERROR: ChromaDB 연결 실패 ({self.chroma_host}:{self.chroma_port}) - {type(e).__name__}: {e}")
            return False

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        ChromaDB 클라이언트와 컬렉션을 초기화하고 LangChain Vectorstore 객체를 반환한다.
        """
        # 구버전 호환을 위해 tenant, database 인수 제거
        chroma_client = HttpClient(
            host=self.chroma_host, # 인스턴스 변수 사용
            port=self.chroma_port, # 인스턴스 변수 사용
        )

        if reset:
            print(f"경고: 기존 컬렉션 '{self.collection_name}'을 삭제하고 새로 만듭니다.")
            
            # 명시적인 컬렉션 리셋 로직을 사용
            try:
                chroma_client.delete_collection(self.collection_name)
                print(f"✅ 컬렉션 '{self.collection_name}' 리셋 완료.")
            except Exception as e:
                print(f"컬렉션 삭제 시도 중 오류 발생 (무시 가능): {e}")
                
        # LangChain Chroma Vectorstore를 명시적으로 생성한 클라이언트와 함께 초기화
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=chroma_client,
            client_settings={"chroma_api_impl": "rest"}
        )
        
        return vectorstore

    def get_retriever(self, k: int = 5) -> Any: 
        """
        설정된 Vectorstore를 기반으로 Retriever 객체를 반환한다.
        """
        vectorstore = self.init_vectorstore(reset=False)
        
        # 유사도 검색(Similarity Search) 기반의 Retriever 반환
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    # ChromaDB 연결 테스트
    print("=" * 50)
    print("VectorDatabaseClient 모듈 테스트 시작")
    print("=" * 50)
    
    from dotenv import load_dotenv
    load_dotenv() 

    # 💡 [수정]: 테스트 환경에 맞춰 8001 포트 사용 강제
    os.environ["CHROMA_HOST"] = "localhost"
    os.environ["CHROMA_PORT"] = "8001"
    
    # 테스트 클라이언트 초기화
    test_client = VectorDatabaseClient(
        collection_name="test_collection",
        embedding_model="solar-embedding-1-large"
    )

    # 1. 헬스 체크
    if test_client.health_check():
        print(f"✅ ChromaDB 연결 성공 (URL: {test_client.chroma_url})")
        
        # 2. 컬렉션 초기화 및 리셋 테스트
        print("\n컬렉션 리셋 테스트 시작...")
        try:
            test_client.init_vectorstore(reset=True)
            print("✅ 컬렉션 리셋 및 초기화 성공")
            
            # 3. Retriever 테스트
            retriever = test_client.get_retriever(k=3)
            print(f"✅ Retriever 생성 성공 (타입: {type(retriever)})")

        except Exception as e:
            print(f"❌ 초기화 및 Retriever 테스트 실패: {e}")
            
    else:
        print(f"❌ ChromaDB 연결 실패. URL: {test_client.chroma_url} 서버가 실행 중인지 확인하세요.")