"""
Docker 컨테이너 환경을 위한 벡터 데이터베이스 모듈
ChromaDB 서버와 연동하는 클라이언트 구현.
"""

import os
import time
from typing import Optional, Any, Dict, List

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 모듈 내부의 llm.py에서 get_embeddings 함수를 가져옴 (상대 경로 사용)
from .llm import get_embeddings


def _is_running_in_docker() -> bool:
    """컨테이너 내부에서 실행 중인지 확인합니다 (로컬/Docker 자동 판별용)."""
    # Docker 환경에서 흔히 발견되는 환경 변수를 확인하거나, 
    # Docker가 생성하는 특정 파일을 확인합니다. (간단한 환경변수 체크 사용)
    return os.path.exists("/.dockerenv") or "DOCKER_CONTAINER_ID" in os.environ


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
        """
        
        # 💡 [핵심 수정]: 실행 환경에 따라 ChromaDB 호스트를 자동으로 결정합니다.
        
        # 1. 환경 변수에서 기본값(vector_db 또는 localhost)을 가져옴
        env_host = os.getenv("CHROMA_HOST", "vector_db") 
        resolved_host = host or env_host

        # 2. 로컬에서 실행 중인데 host가 'vector_db'로 설정되어 있다면, 'localhost'로 변경합니다.
        if not _is_running_in_docker() and resolved_host == "vector_db":
            # 로컬 PC가 Docker 컨테이너(vector_db)에 접속하려면 localhost를 써야 함
            resolved_host = "localhost"

        # 3. 만약 Docker 컨테이너 안에서 실행 중인데 host가 'localhost'라면, 'vector_db'로 재변경합니다.
        #    (이 경우는 거의 없지만, 완전성을 위해 대비)
        elif _is_running_in_docker() and resolved_host == "localhost":
            # 컨테이너 안에서는 'localhost'는 자기 자신을 가리키므로, 옆 컨테이너 이름으로 변경
            resolved_host = "vector_db"

        self.host: str = resolved_host
        # 환경 변수에서 PORT를 가져올 때 int로 변환 (PEP 484)
        self.port: int = port or int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name: str = collection_name

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
        임베딩 모델 선택 및 인스턴스 반환. (PEP 484)
        """
        return get_embeddings(model=model_name)

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        벡터 저장소를 초기화하고 Chroma 인스턴스를 반환.
        """
        # 기존 컬렉션 삭제 (reset=True인 경우)
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"기존 컬렉션 삭제: {self.collection_name}")
            except Exception as e:
                # 컬렉션이 없는 경우의 에러는 무시 (PEP 20)
                print(f"컬렉션 삭제 실패 (이미 없을 수 있음): {e.__class__.__name__}")

        # Chroma 벡터스토어 생성
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        try:
            # 컬렉션의 count를 통해 초기화 여부 및 데이터 수 확인
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            print(f"벡터 저장소 초기화 완료: {self.collection_name}, 현재 문서 수: {count}")
        except Exception:
            # 새로 컬렉션이 생성되었거나, 연결은 되었지만 아직 데이터가 없는 경우
            print(f"벡터 저장소 초기화 완료: 새로운 컬렉션 '{self.collection_name}' 생성됨")

        return self.vectorstore

    def health_check(self) -> bool:
        """
        ChromaDB 서버 헬스체크. (PEP 484)
        """
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            print(f"ChromaDB 서버 연결 실패: {e.__class__.__name__} - {e}")
            return False


# API 서버(main.py)에서 사용할 편의 함수
def get_persisted_vectorstore(
    host: Optional[str] = None,
    collection_name: str = "langchain_docs",
) -> Chroma:
    """
    초기 적재가 완료된 벡터 저장소 객체를 가져옴 (main.py에서 사용될 함수). (PEP 484)
    """
    vdb_client: VectorDatabaseClient = VectorDatabaseClient(
        host=host,
        collection_name=collection_name
    )

    if not vdb_client.health_check():
        # main.py에서 HTTP 500 에러를 던질 수 있도록 ConnectionError를 발생시킴
        raise ConnectionError("ChromaDB 서버에 연결할 수 없습니다. Docker Compose를 확인해주세요.")

    # reset=False로 호출하여 기존 데이터를 유지
    return vdb_client.init_vectorstore(reset=False)


if __name__ == "__main__":
    # 개발 환경에서 .env 로드가 필요할 수 있으므로 여기서 로드 (PEP 8)
    from dotenv import load_dotenv
    load_dotenv()
    
    # 로컬에서 실행할 때는 CHROMA_HOST를 vector_db로 두는 게 편하므로 env 파일에 vector_db를 넣는 게 좋음
    if os.getenv("CHROMA_HOST") is None:
        print("경고: CHROMA_HOST 환경 변수가 설정되지 않아 'vector_db'를 기본값으로 사용합니다.")
        os.environ["CHROMA_HOST"] = "vector_db"

    print("=" * 60)
    print("VectorDatabase 모듈 테스트")
    print("=" * 60)

    try:
        # get_persisted_vectorstore 함수를 사용하여 테스트 (main.py가 사용할 방식)
        vectorstore = get_persisted_vectorstore()
        print(f"✅ ChromaDB 서버 연결 및 VectorStore 로드 성공: {type(vectorstore).__name__}")
        # 임베딩 함수 타입 확인 (Solar Embeddings 사용)
        print(f"✅ 임베딩 모델: {vectorstore.embedding_function.__class__.__name__}")
        
    except ConnectionError as e:
        print(f"❌ [에러 발생]: {e}")
        print("\nDocker Compose로 ChromaDB 컨테이너를 먼저 시작해야 합니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e.__class__.__name__} - {e}")