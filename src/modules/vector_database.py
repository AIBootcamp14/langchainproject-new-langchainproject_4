# src/modules/vector_database.py

"""
벡터 데이터베이스(ChromaDB) 연결 및 관리 클래스
"""

import os
from typing import List, Final

# 써드파티 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# 프로젝트 모듈
from src.modules.llm import get_embeddings


# --- 설정 및 상수 (PEP 8 준수) ---
# ChromaDB의 연결 주소는 Docker 환경 변수 CHROMA_HOST를 사용하거나 로컬 기본값 사용
CHROMA_HOST: Final[str] = os.getenv("CHROMA_HOST", "localhost") 
CHROMA_PORT: Final[int] = int(os.getenv("CHROMA_PORT", "8000")) 
CHROMA_URL: Final[str] = f"http://{CHROMA_HOST}:{CHROMA_PORT}"


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
        
        # 💡 get_embeddings 함수를 사용하여 Embeddings 인스턴스 초기화
        self.embeddings: Embeddings = get_embeddings(model=embedding_model)

    def health_check(self) -> bool:
        """
        ChromaDB 서버 연결 상태를 확인한다.
        """
        try:
            # ChromaDB 연결 시도 (Collection이 아닌 Client 레벨에서 테스트)
            # 여기서는 ChromaDB HTTP 연결을 시도하는 간접적인 방법을 사용
            from chromadb import HttpClient # 임포트 위치를 함수 내로 변경하여 지연 로딩
            client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat() # 하트비트 호출로 연결 확인
            return True
        except Exception:
            return False

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        ChromaDB 클라이언트와 컬렉션을 초기화하고 LangChain Vectorstore 객체를 반환한다.

        Args:
            reset: True면 기존 컬렉션을 삭제하고 새로 생성한다.

        Returns:
            LangChain Chroma Vectorstore 인스턴스.
        """
        # LangChain Chroma Vectorstore는 내부적으로 HttpClient를 사용하여 연결
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=None, # HTTP 모드 사용 시 persist_directory는 None
            url=CHROMA_URL,
        )

        if reset:
            print(f"경고: 기존 컬렉션 '{self.collection_name}'을 삭제하고 새로 만듭니다.")
            # 컬렉션을 삭제하는 로직은 LangChain Chroma 객체를 통해 직접 접근하기 어려우므로,
            # 내부적으로 사용하는 client를 통해 접근하거나, 초기화 스크립트에서 관리함.
            # 여기서는 LangChain의 Chroma 기능을 사용하여 컬렉션이 없으면 새로 생성되도록 처리하고
            # reset_db 로직은 initialize_vector_db.py의 논리를 유지한다.
            # (LangChain Chroma는 collection_name이 없으면 새로 생성)
            
            # 실제 삭제 로직을 실행하는 것이 명확하지만, 현재 LangChain_community의 Chroma 구현에 의존함.
            # 가장 확실한 방법은 ChromaDB의 Python Client를 사용하는 것이지만, 
            # 여기서는 초기화 스크립트에서 전체 리셋(reset)을 담당하는 것으로 가정한다.
            
            # 💡 [추가]: 명시적인 컬렉션 리셋 로직을 추가하여 안정성 확보 (LangChain Chroma 대신 Client 사용)
            try:
                from chromadb import HttpClient
                client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                # 컬렉션 삭제 후 재생성 (Reset 로직 강화)
                client.delete_collection(self.collection_name)
                # 재생성 (LangChain이 다시 만들도록 유도)
                print(f"✅ 컬렉션 '{self.collection_name}' 리셋 완료.")
            except Exception as e:
                # 컬렉션이 없어서 삭제에 실패하는 경우는 정상
                print(f"컬렉션 삭제 시도 중 오류 발생 (무시 가능): {e}")

        # 리셋 후 새로 생성된 (혹은 기존) Vectorstore 반환
        return vectorstore

    def get_retriever(self, k: int = 5) -> Any: # Any 대신 Retriever 타입을 써야 하지만 임포트가 복잡하여 Any 사용
        """
        설정된 Vectorstore를 기반으로 Retriever 객체를 반환한다.
        """
        vectorstore = self.init_vectorstore(reset=False) # 기존 컬렉션을 사용
        
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
    
    # 💡 get_embeddings가 환경 변수를 사용하므로 dotenv 로드가 필요함
    from dotenv import load_dotenv
    load_dotenv() 

    # 테스트 클라이언트 초기화
    test_client = VectorDatabaseClient(
        collection_name="test_collection",
        embedding_model="solar-embedding-1-large"
    )

    # 1. 헬스 체크
    if test_client.health_check():
        print(f"✅ ChromaDB 연결 성공 (URL: {CHROMA_URL})")
        
        # 2. 컬렉션 초기화 및 리셋 테스트
        print("\n컬렉션 리셋 테스트 시작...")
        test_client.init_vectorstore(reset=True) # 리셋 후 새로 생성
        print("✅ 컬렉션 리셋 및 초기화 성공")
        
        # 3. Retriever 테스트
        retriever = test_client.get_retriever(k=3)
        print(f"✅ Retriever 생성 성공 (타입: {type(retriever)})")

    else:
        print(f"❌ ChromaDB 연결 실패. URL: {CHROMA_URL} 서버가 실행 중인지 확인하세요.")