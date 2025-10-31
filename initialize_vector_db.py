"""
Vector DB 초기화 스크립트
LangChain 문서를 수집하고 구조 기반 청킹 후 Vector DB에 적재합니다.
"""

import os
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# 써드파티 라이브러리
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # 임시 스플리터

# 프로젝트 모듈 임포트 (루트에서 실행되므로 src.으로 시작)
from src.modules.vector_database import VectorDatabaseClient
# from src.data_collector import DataCollector # 👈 팀원 4의 DataCollector 클래스로 대체 필요
# from src.utils.chunking_strategy import StructuredTextSplitter # 👈 팀원 4의 청킹 전략으로 대체 필요


# 임시 DataCollector 클래스 (팀원 4 파일 올 때까지 사용)
class DummyDataCollector:
    """팀원 4의 DataCollector가 오기 전까지 사용하는 더미 클래스"""
    def collect_documents(self, urls: List[str], max_pages: int, delay: float) -> List[Document]:
        print("⚠️ 더미 DataCollector 사용: 실제 문서를 수집하지 않습니다.")
        # 더미 문서 반환
        return [
            Document(page_content=f"더미 문서 {i}: 이 문서는 {url}에서 왔습니다.", metadata={"source": url, "title": f"Doc {i}"})
            for i, url in enumerate(urls[:max_pages])
        ]

# 임시 TextSplitter 클래스 (팀원 4 파일 올 때까지 사용)
class DummyTextSplitter(RecursiveCharacterTextSplitter):
    """팀원 4의 StructuredTextSplitter가 오기 전까지 사용하는 더미 클래스"""
    def split_documents(self, documents: List[Document]) -> List[Document]:
        print("⚠️ 더미 TextSplitter 사용: RecursiveCharacterTextSplitter를 사용합니다.")
        return super().split_documents(documents)

# LangChain 문서 URL 목록 (확장 가능)
LANGCHAIN_URLS: List[str] = [
    # 핵심 개념
    "https://python.langchain.com/docs/introduction",
    "https://python.langchain.com/docs/get_started/quickstart",
    # 주요 모듈
    "https://python.langchain.com/docs/modules/model_io",
    "https://python.langchain.com/docs/modules/data_connection",
    "https://python.langchain.com/docs/modules/chains",
    "https://python.langchain.com/docs/modules/agents",
    "https://python.langchain.com/docs/modules/memory",
]


class VectorDBInitializer:
    """Vector DB 초기화 및 데이터 로딩 클래스"""

    def __init__(
        self,
        docker_host: str = "localhost",
        docker_port: int = 8000,
        collection_name: str = "langchain_docs",
        embedding_model: str = "solar-embedding-1-large",
    ) -> None:
        """
        초기화 (Docker ChromaDB 전용으로 단순화)
        """
        print(f"🐳 Docker ChromaDB 사용: {docker_host}:{docker_port}")
        
        # Vector DB 클라이언트 초기화 (통합된 VectorDatabaseClient 사용)
        self.vector_db = VectorDatabaseClient(
            host=docker_host,
            port=docker_port,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        # 데이터 수집기 (더미 사용)
        self.collector = DummyDataCollector() # 👈 나중에 DataCollector로 교체

        # 구조 기반 텍스트 분할기 (더미 사용)
        # ⚠️ 팀원 4의 StructuredTextSplitter로 교체 예정
        self.text_splitter = DummyTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    def collect_documents(
        self,
        urls: Optional[List[str]] = None,
        max_pages: int = 50,
    ) -> List[Document]:
        """문서 수집"""
        print("\n📥 문서 수집 시작...")

        if urls is None:
            urls = LANGCHAIN_URLS

        documents = self.collector.collect_documents(
            urls=urls[:max_pages],
            max_pages=max_pages,
            delay=1.0,  # 서버 부하 방지
        )

        print(f"✅ {len(documents)}개 문서 수집 완료")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """구조 기반 문서 청킹"""
        print("\n✂️ 구조 기반 청킹 시작...")

        # ⚠️ 팀원 4의 로직을 사용할 경우 통계 계산 필요
        chunked_docs = self.text_splitter.split_documents(documents)

        print(f"✅ 청킹 완료: 총 청크 {len(chunked_docs)}개")
        return chunked_docs

    def load_to_vector_db(
        self,
        documents: List[Document],
        batch_size: int = 100,
        reset: bool = False,
    ) -> int:
        """문서를 Vector DB에 적재"""
        print("\n💾 Vector DB 적재 시작...")

        # Docker ChromaDB 헬스체크
        if not self.vector_db.health_check():
            raise ConnectionError(
                "ChromaDB 서버에 연결할 수 없습니다. "
                "Docker Compose 실행: docker-compose up -d chromadb"
            )

        # 벡터스토어 초기화
        self.vector_db.init_vectorstore(reset=reset)

        # 문서 추가
        ids = self.vector_db.add_documents(
            documents=documents,
            batch_size=batch_size,
            show_progress=True,
        )

        print(f"✅ {len(ids)}개 문서 Vector DB 적재 완료")
        return len(ids)

    def run_full_pipeline(
        self,
        urls: Optional[List[str]] = None,
        max_pages: int = 30,
        reset_db: bool = False,
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        print("=" * 60)
        print("🚀 Vector DB 초기화 파이프라인 시작")
        print("=" * 60)

        try:
            # 1. 문서 수집
            documents = self.collect_documents(urls, max_pages)

            # 2. 구조 기반 청킹
            chunked_docs = self.chunk_documents(documents)

            # 3. Vector DB 적재
            loaded_count = self.load_to_vector_db(
                chunked_docs,
                batch_size=100,
                reset=reset_db,
            )

            # 4. 테스트 검색
            print("\n🔍 테스트 검색...")
            test_query = "LangChain에서 메모리를 사용하는 방법"
            results = self.vector_db.search(test_query, k=1) # 1개만 테스트
            print(f"검색 결과: {len(results)}개 문서")

            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()

            # 결과 반환
            result = {
                "status": "success",
                "documents_collected": len(documents),
                "chunks_created": len(chunked_docs),
                "documents_loaded": loaded_count,
                "execution_time": f"{execution_time:.2f}초",
                "vector_db_type": "Docker ChromaDB",
            }

            print("\n" + "=" * 60)
            print("✅ 파이프라인 완료!")
            print("=" * 60)
            print(f"총 실행 시간: {result['execution_time']}")

            return result

        except Exception as e:
            print(f"\n❌ 파이프라인 실행 실패: {e}")
            return {"status": "failed", "error": str(e)}


def main() -> None:
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Vector DB 초기화 및 데이터 로딩"
    )

    # Docker 설정은 기본값으로 고정 (프로젝트 기준)
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("CHROMA_HOST", "localhost"), # ENV 변수 우선 사용
        help="Docker ChromaDB 호스트 (기본: CHROMA_HOST 환경 변수 또는 localhost)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="langchain_docs",
        help="컬렉션 이름 (기본: langchain_docs)"
    )

    parser.add_argument(
        "--embedding",
        type=str,
        default="solar-embedding-1-large",
        # 선택지 정리
        choices=[
            "solar-embedding-1-large",
            "solar-embedding-1-large-query",
            "solar-embedding-1-large-passage",
            "ko-sbert-multitask"
        ],
        help="임베딩 모델 (기본: solar-embedding-1-large)"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=len(LANGCHAIN_URLS), # 기본값: 전체 URL 수
        help="최대 수집 페이지 수 (기본: 전체)"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재생성"
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="테스트 모드 (일부 문서만)"
    )

    args = parser.parse_args()

    # 초기화
    initializer = VectorDBInitializer(
        docker_host=args.host,
        collection_name=args.collection,
        embedding_model=args.embedding
    )

    # 테스트 모드
    if args.test_only:
        print("🧪 테스트 모드: 일부 문서만 처리")
        urls = LANGCHAIN_URLS[:3]
        max_pages = 3
    else:
        urls = LANGCHAIN_URLS
        max_pages = args.max_pages

    # 파이프라인 실행
    initializer.run_full_pipeline(
        urls=urls,
        max_pages=max_pages,
        reset_db=args.reset
    )


if __name__ == "__main__":
    main()