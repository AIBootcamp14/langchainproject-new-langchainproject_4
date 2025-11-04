"""
벡터 데이터베이스(ChromaDB) 연결 및 관리 클래스
"""

from typing import Any, Final
import os
from typing import List

# 써드파티 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
# 💡 [핵심] Chroma 클라이언트 명시적 사용을 위해 최상단 임포트
from chromadb import HttpClient 


# --- 설정 및 상수 (PEP 8 준수) ---
CHROMA_HOST: Final[str] = os.getenv("CHROMA_HOST", "localhost") 
CHROMA_PORT: Final[int] = int(os.getenv("CHROMA_PORT", "8000")) 
CHROMA_URL: Final[str] = f"http://{CHROMA_HOST}:{CHROMA_PORT}"
# 💡 [추가] 명시적인 Tenant/Database 설정
CHROMA_TENANT: Final[str] = "default_tenant"
CHROMA_DATABASE: Final[str] = "default_database"


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
        from src.modules.llm import get_embeddings
        self.embeddings: Embeddings = get_embeddings(model=embedding_model)

    def health_check(self) -> bool:
        """
        ChromaDB 서버 연결 상태를 확인한다.
        """
        # ❌ [FATAL ERROR CHECK] 이 코드가 없으면 최신 코드가 아님을 알림
        if CHROMA_TENANT != "default_tenant":
            print("FATAL_CODE_ERROR: 'src/modules/vector_database.py' 파일이 최신 버전으로 복사되지 않았습니다! (tenant 명시 누락)")
            return False # 연결 시도 중단
        
        try:
            # 💡 [수정] HttpClient에 tenant 및 database 인수를 명시적으로 전달하여 연결 안정화
            client = HttpClient(
                host=CHROMA_HOST, 
                port=CHROMA_PORT,
                tenant=CHROMA_TENANT,  # 기본값 명시
                database=CHROMA_DATABASE # 기본값 명시
            )
            client.heartbeat() # 하트비트 호출로 연결 확인
            return True
        except Exception as e: 
            # ❌ [디버그 코드 유지]: 연결 실패 시 오류 출력
            print(f"DEBUG_CHROMA_ERROR: ChromaDB 연결 실패 ({CHROMA_HOST}:{CHROMA_PORT}) - {type(e).__name__}: {e}")
            return False

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        ChromaDB 클라이언트와 컬렉션을 초기화하고 LangChain Vectorstore 객체를 반환한다.
        """
        # 💡 [핵심 수정]: ChromaDB 클라이언트를 명시적으로 생성하여 LangChain에 전달
        chroma_client = HttpClient(
            host=CHROMA_HOST, 
            port=CHROMA_PORT,
            tenant=CHROMA_TENANT, 
            database=CHROMA_DATABASE
        )

        if reset:
            print(f"경고: 기존 컬렉션 '{self.collection_name}'을 삭제하고 새로 만듭니다.")
            
            # 💡 [개선] 명시적인 컬렉션 리셋 로직을 사용
            try:
                chroma_client.delete_collection(self.collection_name)
                print(f"✅ 컬렉션 '{self.collection_name}' 리셋 완료.")
            except Exception as e:
                # 컬렉션이 없어서 삭제에 실패하는 경우는 정상
                print(f"컬렉션 삭제 시도 중 오류 발생 (무시 가능): {e}")
                
        # LangChain Chroma Vectorstore를 명시적으로 생성한 클라이언트와 함께 초기화
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=chroma_client, # 💡 [핵심] 명시적으로 생성한 클라이언트 객체를 전달
            client_settings={"chroma_api_impl": "rest"} # API 구현 방식을 명시
        )
        
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
        print(f"✅ ChromaDB 연결 성공 (URL: {CHROMA_URL}, Tenant: {CHROMA_TENANT})")
        
        # 2. 컬렉션 초기화 및 리셋 테스트
        print("\n컬렉션 리셋 테스트 시작...")
        # 💡 init_vectorstore이 성공적으로 클라이언트를 넘겨주는지 확인
        try:
             test_client.init_vectorstore(reset=True) # 리셋 후 새로 생성
             print("✅ 컬렉션 리셋 및 초기화 성공")
             
             # 3. Retriever 테스트
             retriever = test_client.get_retriever(k=3)
             print(f"✅ Retriever 생성 성공 (타입: {type(retriever)})")

        except Exception as e:
            print(f"❌ 초기화 및 Retriever 테스트 실패: {e}")
            
    else:
        print(f"❌ ChromaDB 연결 실패. URL: {CHROMA_URL} 서버가 실행 중인지 확인하세요.")