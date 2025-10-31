# src/modules/vector_database.py

"""
Docker 컨테이너 환경을 위한 벡터 데이터베이스 모듈
ChromaDB 서버와 연동하는 클라이언트 구현.
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 모듈 내부의 llm.py에서 get_embeddings 함수를 가져옴 (상대 경로 사용)
from .llm import get_embeddings


class VectorDatabaseClient:
    """
    Docker ChromaDB 서버와 연동하는 벡터 데이터베이스 클라이언트 클래스.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 8000,
        collection_name: str = "langchain_docs",
        embedding_model: str = "solar-embedding-1-large",
    ) -> None:
        """
        VectorDatabaseClient 초기화.

        Args:
            host: ChromaDB 서버 호스트 (기본값: 환경변수 CHROMA_HOST 또는 localhost).
            port: ChromaDB 서버 포트 (기본값: 8000).
            collection_name: 컬렉션 이름.
            embedding_model: 사용할 임베딩 모델 이름.
        """
        # 호스트 설정 수정: 환경변수 CHROMA_HOST 값이 'vector_db'로 설정되어 있으면 
        # 로컬 실행 시 강제로 'localhost'로 변경해 줘야 해.
        resolved_host = host or os.getenv("CHROMA_HOST", "localhost")
        
        # 🌟🌟🌟 이 조건문을 추가해 줘 🌟🌟🌟
        if resolved_host == "vector_db":
            resolved_host = "localhost" # 로컬 환경에서 실행할 경우 'vector_db' 대신 'localhost' 사용

        self.host = resolved_host
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = collection_name

        # 임베딩 모델 설정
        self.embeddings: Embeddings = self._get_embedding_model(embedding_model)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.vectorstore: Optional[Chroma] = None

        print(f"ChromaDB 서버 연결 시도: {self.host}:{self.port}")

    def _get_embedding_model(self, model_name: str) -> Embeddings:
        """
        임베딩 모델 선택 및 인스턴스 반환. (타입 힌트 강화: PEP 484)
        """
        # Solar Embedding이 확정되었으므로, 다른 모델은 옵션으로 남겨둠
        return get_embeddings(model=model_name)

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        벡터 저장소를 초기화하고 Chroma 인스턴스를 반환.
        RAG 체인(retriever.py)에 필요한 VectorStore 객체를 제공하는 핵심 함수.

        Args:
            reset: True일 경우 기존 컬렉션 삭제 후 재생성.

        Returns:
            Chroma: 초기화된 Chroma 벡터 저장소 인스턴스.
        """
        # 기존 컬렉션 삭제 (reset=True인 경우)
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"기존 컬렉션 삭제: {self.collection_name}")
            except Exception as e:
                print(f"컬렉션 삭제 실패 (없을 수 있음): {e}")

        # Chroma 벡터스토어 생성
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            print(f"벡터 저장소 초기화 완료: {self.collection_name}, 현재 문서 수: {count}")
        except:
            print(f"벡터 저장소 초기화 완료: 새로운 컬렉션 '{self.collection_name}' 생성됨")

        return self.vectorstore
    
    def health_check(self) -> bool:
        """
        ChromaDB 서버 헬스체크.
        """
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            print(f"ChromaDB 서버 연결 실패: {e}")
            return False

    # add_documents, search, get_statistics, backup_collection 등은
    # main.py 통합에는 불필요하므로 팀원 4, 5의 데이터 적재 스크립트에 포함시키거나 
    # 별도 유틸리티로 관리하도록 남겨둠. 여기서는 핵심만 남김.


# API 서버(main.py)에서 사용할 편의 함수
def get_persisted_vectorstore(
    host: Optional[str] = None,
    collection_name: str = "langchain_docs",
) -> Chroma:
    """
    초기 적재가 완료된 벡터 저장소 객체를 가져옴 (main.py에서 사용될 함수).
    """
    vdb_client = VectorDatabaseClient(
        host=host,
        collection_name=collection_name
    )

    if not vdb_client.health_check():
        raise ConnectionError("ChromaDB 서버에 연결할 수 없습니다. Docker Compose를 확인해주세요.")

    # reset=False로 호출하여 기존 데이터를 유지
    return vdb_client.init_vectorstore(reset=False)


if __name__ == "__main__":
    # ChromaDB 연결 테스트 및 초기화 테스트
    print("=" * 60)
    print("VectorDatabase 모듈 테스트")
    print("=" * 60)

    try:
        # get_persisted_vectorstore 함수를 사용하여 테스트 (main.py가 사용할 방식)
        vectorstore = get_persisted_vectorstore()
        print(f"✅ ChromaDB 서버 연결 및 VectorStore 로드 성공: {type(vectorstore)}")
        # 임베딩 함수 타입 확인 (Solar Embeddings 사용)
        print(f"✅ 임베딩 모델: {vectorstore.embedding_function.__class__.__name__}")
        
    except ConnectionError as e:
        print(f"❌ [에러 발생]: {e}")
        print("\nDocker Compose로 ChromaDB 컨테이너를 먼저 시작해야 합니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")