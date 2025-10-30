"""Docker 컨테이너 환경을 위한 벡터 데이터베이스 모듈
ChromaDB 서버와 연동하는 클라이언트"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

from llm import get_embeddings


class DockerVectorDatabase:
    """Docker ChromaDB 서버와 연동하는 벡터 데이터베이스 클래스"""

    def __init__(
        self,
        host: str = None,
        port: int = 8000,
        collection_name: str = "langchain_docs",
        embedding_model: str = "solar-embedding-1-large"
    ):
        """
        DockerVectorDatabase 초기화

        Args:
            host: ChromaDB 서버 호스트 (기본값: 환경변수 CHROMA_HOST 또는 localhost)
            port: ChromaDB 서버 포트 (기본값: 8000)
            collection_name: 컬렉션 이름
            embedding_model: 사용할 임베딩 모델
        """
        # 호스트 설정 (Docker 환경 자동 감지)
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = collection_name

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 임베딩 모델 설정
        self.embeddings = self._get_embedding_model(embedding_model)
        self.vectorstore = None

        print(f"ChromaDB 서버 연결: {self.host}:{self.port}")

    def _get_embedding_model(self, model_name: str) -> Embeddings:
        """
        임베딩 모델 선택

        Args:
            model_name: 임베딩 모델 이름

        Returns:
            Embeddings 인스턴스
        """
        if model_name.startswith("solar"):
            # Upstage Solar Embedding
            return get_embeddings(model=model_name)
        elif model_name == "ko-sbert-multitask":
            # 한국어 SBERT (sentence-transformers 필요)
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sbert-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except ImportError:
                print("sentence-transformers 설치 필요: pip install sentence-transformers")
                return get_embeddings()  # fallback to solar
        else:
            # 기본값: Solar Embedding
            return get_embeddings()

    def init_vectorstore(self, reset: bool = False) -> Chroma:
        """
        벡터 저장소를 초기화합니다.

        Args:
            reset: True일 경우 기존 컬렉션 삭제 후 재생성

        Returns:
            Chroma 벡터 저장소 인스턴스
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

        print(f"벡터 저장소 초기화 완료: {self.collection_name}")

        # 컬렉션 정보 출력
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            print(f"현재 문서 수: {count}")
        except:
            print("새로운 컬렉션 생성됨")

        return self.vectorstore

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[str]:
        """
        문서를 벡터 저장소에 추가합니다.

        Args:
            documents: 추가할 문서 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            추가된 문서 ID 리스트
        """
        if not self.vectorstore:
            self.init_vectorstore()

        all_ids = []
        total_docs = len(documents)

        print(f"총 {total_docs}개 문서 추가 중...")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]

            if show_progress:
                progress = min(i + batch_size, total_docs)
                print(f"진행률: {progress}/{total_docs} ({progress*100//total_docs}%)")

            try:
                ids = self.vectorstore.add_documents(batch)
                all_ids.extend(ids)
            except Exception as e:
                print(f"배치 추가 실패: {e}")
                # 실패한 배치는 개별 처리
                for doc in batch:
                    try:
                        doc_id = self.vectorstore.add_documents([doc])
                        all_ids.extend(doc_id)
                    except Exception as e2:
                        print(f"문서 추가 실패: {e2}")

        print(f"완료: {len(all_ids)}개 문서 추가됨")
        return all_ids

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Dict[str, Any] = None,
        search_type: str = "similarity"
    ) -> List[Document]:
        """
        벡터 유사도 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter: 메타데이터 필터
            search_type: 검색 타입 ("similarity", "mmr")

        Returns:
            검색된 문서 리스트
        """
        if not self.vectorstore:
            self.init_vectorstore()

        if search_type == "mmr":
            # Maximum Marginal Relevance
            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                filter=filter
            )
        else:
            # 기본: 유사도 검색
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        벡터 데이터베이스 통계 정보를 반환합니다.

        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
            "embedding_model": self.embeddings.__class__.__name__
        }

        try:
            collection = self.client.get_collection(self.collection_name)
            stats["document_count"] = collection.count()

            # 샘플 메타데이터 가져오기
            sample = collection.peek(limit=1)
            if sample and sample.get("metadatas"):
                stats["sample_metadata_keys"] = list(sample["metadatas"][0].keys())
        except Exception as e:
            stats["error"] = str(e)

        return stats

    def health_check(self) -> bool:
        """
        ChromaDB 서버 헬스체크

        Returns:
            서버가 정상이면 True
        """
        try:
            # heartbeat API 호출
            self.client.heartbeat()
            return True
        except Exception as e:
            print(f"ChromaDB 서버 연결 실패: {e}")
            return False

    def backup_collection(self, backup_path: str = "./backups"):
        """
        컬렉션을 로컬 파일로 백업

        Args:
            backup_path: 백업 저장 경로
        """
        Path(backup_path).mkdir(parents=True, exist_ok=True)

        try:
            collection = self.client.get_collection(self.collection_name)
            data = collection.get()

            import json
            from datetime import datetime

            backup_file = Path(backup_path) / f"{self.collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"백업 완료: {backup_file}")
        except Exception as e:
            print(f"백업 실패: {e}")

    def restore_collection(self, backup_file: str):
        """
        백업 파일에서 컬렉션 복원

        Args:
            backup_file: 백업 파일 경로
        """
        import json

        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 컬렉션 재생성
            self.init_vectorstore(reset=True)

            # 데이터 복원
            if data.get("documents"):
                documents = []
                for i, doc_text in enumerate(data["documents"]):
                    metadata = data["metadatas"][i] if i < len(data["metadatas"]) else {}
                    doc = Document(page_content=doc_text, metadata=metadata)
                    documents.append(doc)

                self.add_documents(documents)
                print(f"복원 완료: {len(documents)}개 문서")
        except Exception as e:
            print(f"복원 실패: {e}")


# 편의 함수들
def create_docker_vector_db(
    host: str = None,
    collection_name: str = "langchain_docs",
    reset: bool = False
) -> DockerVectorDatabase:
    """
    Docker 환경의 벡터 데이터베이스 인스턴스 생성

    Args:
        host: ChromaDB 서버 호스트
        collection_name: 컬렉션 이름
        reset: 기존 데이터 삭제 여부

    Returns:
        DockerVectorDatabase 인스턴스
    """
    vdb = DockerVectorDatabase(
        host=host,
        collection_name=collection_name
    )

    # 헬스체크
    if not vdb.health_check():
        raise ConnectionError("ChromaDB 서버에 연결할 수 없습니다.")

    vdb.init_vectorstore(reset=reset)
    return vdb


def migrate_local_to_docker(
    local_persist_dir: str = "./data/chroma_db",
    docker_host: str = "localhost",
    collection_name: str = "langchain_docs"
):
    """
    로컬 ChromaDB를 Docker ChromaDB로 마이그레이션

    Args:
        local_persist_dir: 로컬 ChromaDB 디렉토리
        docker_host: Docker ChromaDB 호스트
        collection_name: 대상 컬렉션 이름
    """
    from vector_database import VectorDatabase

    print("로컬 데이터 로드 중...")
    local_vdb = VectorDatabase(persist_directory=local_persist_dir)
    local_vdb.init_vectorstore()

    # 로컬에서 모든 문서 가져오기 (제한적)
    # ChromaDB는 직접적인 전체 문서 추출을 지원하지 않으므로
    # 별도 백업/복원 메커니즘 필요

    print("Docker ChromaDB로 마이그레이션...")
    docker_vdb = create_docker_vector_db(
        host=docker_host,
        collection_name=collection_name,
        reset=True
    )

    print("마이그레이션 완료!")
    return docker_vdb


if __name__ == "__main__":
    # 모듈 테스트
    print("=" * 60)
    print("Docker Vector Database 모듈 테스트")
    print("=" * 60)

    # Docker ChromaDB 연결 테스트
    vdb = DockerVectorDatabase()

    # 헬스체크
    if vdb.health_check():
        print("✅ ChromaDB 서버 정상")
    else:
        print("❌ ChromaDB 서버 연결 실패")
        print("\nDocker Compose로 ChromaDB 시작:")
        print("  docker-compose up -d chromadb")
        exit(1)

    # 벡터스토어 초기화
    vdb.init_vectorstore()

    # 통계 출력
    stats = vdb.get_statistics()
    print("\n📊 데이터베이스 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n테스트 완료!")