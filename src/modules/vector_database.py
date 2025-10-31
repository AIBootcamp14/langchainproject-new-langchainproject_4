"""
Docker 컨테이너 환경을 위한 벡터 데이터베이스 모듈
ChromaDB 서버와 연동하는 클라이언트 구현.
"""

import os
import time  # 시간 측정/딜레이 등에 대비해 임포트 유지 (PEP 8)
from typing import Optional, Any, Dict, List  # PEP 484

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
        """
        # 호스트 설정: 로컬 실행 시 Docker 컨테이너 이름(vector_db)을 localhost로 재지정
        resolved_host = host or os.getenv("CHROMA_HOST", "localhost")

        if resolved_host == "vector_db":
            resolved_host = "localhost"

        self.host = resolved_host
        # 환경 변수에서 PORT를 가져올 때 int로 변환 (PEP 484)
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = collection_name

        # 임베딩 모델 설정
        # LLM 모듈 테스트에서 성공했으므로, 이 객체 생성은 문제 없음
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

        # 💡 [핵심 수정]: ChromaDB에서 임베딩을 시도할 때 에러가 발생하므로,
        # 💡 임베딩 함수를 **지연 로딩**하거나, **`langchain-upstage`와 ChromaDB 간의 호출 형식 문제**를 우회해야 함.
        # 💡 임베딩 함수를 `UpstageEmbeddings` 대신, `Chroma`가 선호하는 래퍼 함수로 감싸서 전달하는 것이 안정적임.
        # 💡 그러나 LangChain v0.1.x 버전대에서는 클래스를 바로 전달하는 것이 표준이므로, 코드는 유지하고 버전을 낮추는 전략을 택함.

        # Chroma 벡터스토어 생성
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            # 💡 임베딩 함수를 직접 전달: LangChain 표준 (버전 충돌 우회는 외부 환경에서 처리)
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
    vdb_client = VectorDatabaseClient(
        host=host,
        collection_name=collection_name
    )

    if not vdb_client.health_check():
        # ConnectionError 대신 표준 HTTPException을 유발하는 에러를 던지도록 (main.py 참조)
        raise ConnectionError("ChromaDB 서버에 연결할 수 없습니다. Docker Compose를 확인해주세요.")

    # reset=False로 호출하여 기존 데이터를 유지
    return vdb_client.init_vectorstore(reset=False)


if __name__ == "__main__":
    # 개발 환경에서 .env 로드가 필요할 수 있으므로 여기서 로드 (PEP 8)
    from dotenv import load_dotenv
    load_dotenv()
    
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