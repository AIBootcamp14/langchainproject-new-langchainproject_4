#!/usr/bin/env python3
"""
Vector DB 초기화 스크립트
LangChain 문서를 수집하고 구조 기반 청킹 후 Vector DB에 적재
"""

import os
import sys
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# 프로젝트 모듈 임포트
from data_collector import LangChainDataCollector
from advanced_text_splitter import StructuredTextSplitter, create_smart_splitter
from vector_database import VectorDatabase
from vector_database_docker import DockerVectorDatabase, create_docker_vector_db
from langchain.schema import Document


# LangChain 문서 URL 목록 (확장 가능)
LANGCHAIN_URLS = [
    # 핵심 개념
    "https://python.langchain.com/docs/introduction",
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/get_started/quickstart",
    "https://python.langchain.com/docs/get_started/installation",

    # 주요 모듈
    "https://python.langchain.com/docs/modules/model_io",
    "https://python.langchain.com/docs/modules/model_io/llms",
    "https://python.langchain.com/docs/modules/model_io/chat",
    "https://python.langchain.com/docs/modules/model_io/prompts",
    "https://python.langchain.com/docs/modules/model_io/output_parsers",

    # 데이터 연결
    "https://python.langchain.com/docs/modules/data_connection",
    "https://python.langchain.com/docs/modules/data_connection/document_loaders",
    "https://python.langchain.com/docs/modules/data_connection/document_transformers",
    "https://python.langchain.com/docs/modules/data_connection/text_embedding",
    "https://python.langchain.com/docs/modules/data_connection/vectorstores",
    "https://python.langchain.com/docs/modules/data_connection/retrievers",

    # 체인
    "https://python.langchain.com/docs/modules/chains",
    "https://python.langchain.com/docs/modules/chains/foundational/llm_chain",
    "https://python.langchain.com/docs/modules/chains/foundational/sequential_chains",

    # 메모리
    "https://python.langchain.com/docs/modules/memory",
    "https://python.langchain.com/docs/modules/memory/types/buffer",
    "https://python.langchain.com/docs/modules/memory/types/summary",

    # 에이전트
    "https://python.langchain.com/docs/modules/agents",
    "https://python.langchain.com/docs/modules/agents/agent_types",
    "https://python.langchain.com/docs/modules/agents/tools",

    # 사용 사례
    "https://python.langchain.com/docs/use_cases/question_answering",
    "https://python.langchain.com/docs/use_cases/chatbots",
    "https://python.langchain.com/docs/use_cases/summarization",
]


class VectorDBInitializer:
    """Vector DB 초기화 및 데이터 로딩 클래스"""

    def __init__(
        self,
        use_docker: bool = False,
        docker_host: str = "localhost",
        docker_port: int = 8000,
        collection_name: str = "langchain_docs",
        embedding_model: str = "solar-embedding-1-large"
    ):
        """
        초기화

        Args:
            use_docker: Docker ChromaDB 사용 여부
            docker_host: Docker ChromaDB 호스트
            docker_port: Docker ChromaDB 포트
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델
        """
        self.use_docker = use_docker
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Vector DB 설정
        if use_docker:
            print(f"🐳 Docker ChromaDB 사용: {docker_host}:{docker_port}")
            self.vector_db = DockerVectorDatabase(
                host=docker_host,
                port=docker_port,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
        else:
            print("💾 로컬 ChromaDB 사용")
            self.vector_db = VectorDatabase(
                persist_directory="./data/chroma_db",
                collection_name=collection_name
            )

        # 데이터 수집기
        self.collector = LangChainDataCollector()

        # 구조 기반 텍스트 분할기
        self.text_splitter = StructuredTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            code_block_max_size=3000,
            preserve_code_blocks=True,
            preserve_functions=True,
            preserve_markdown_structure=True
        )

    def collect_documents(
        self,
        urls: List[str] = None,
        max_pages: int = 50
    ) -> List[Document]:
        """
        문서 수집

        Args:
            urls: 수집할 URL 목록
            max_pages: 최대 수집 페이지 수

        Returns:
            수집된 문서 리스트
        """
        print("\n📥 문서 수집 시작...")

        if urls is None:
            urls = LANGCHAIN_URLS

        # 문서 수집
        documents = self.collector.collect_documents(
            urls=urls[:max_pages],
            max_pages=max_pages,
            delay=1.0  # 서버 부하 방지
        )

        print(f"✅ {len(documents)}개 문서 수집 완료")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        구조 기반 문서 청킹

        Args:
            documents: 원본 문서 리스트

        Returns:
            청킹된 문서 리스트
        """
        print("\n✂️ 구조 기반 청킹 시작...")

        chunked_docs = []
        code_block_count = 0
        section_count = 0

        for doc in documents:
            # 구조 기반 분할
            chunks = self.text_splitter.split_documents([doc])

            for chunk in chunks:
                # 메타데이터 통계 수집
                if chunk.metadata.get('chunk_type') == 'code':
                    code_block_count += 1
                if chunk.metadata.get('section_title'):
                    section_count += 1

                chunked_docs.append(chunk)

        print(f"✅ 청킹 완료:")
        print(f"  - 총 청크: {len(chunked_docs)}개")
        print(f"  - 코드 블록: {code_block_count}개")
        print(f"  - 섹션: {section_count}개")

        return chunked_docs

    def load_to_vector_db(
        self,
        documents: List[Document],
        batch_size: int = 100,
        reset: bool = False
    ) -> int:
        """
        문서를 Vector DB에 적재

        Args:
            documents: 적재할 문서 리스트
            batch_size: 배치 크기
            reset: 기존 데이터 삭제 여부

        Returns:
            적재된 문서 수
        """
        print("\n💾 Vector DB 적재 시작...")

        # Vector DB 초기화
        if self.use_docker:
            # Docker ChromaDB 헬스체크
            if not self.vector_db.health_check():
                raise ConnectionError(
                    "ChromaDB 서버에 연결할 수 없습니다.\n"
                    "Docker Compose 실행: docker-compose up -d chromadb"
                )

        # 벡터스토어 초기화
        self.vector_db.init_vectorstore(reset=reset)

        # 문서 추가
        ids = self.vector_db.add_documents(
            documents=documents,
            batch_size=batch_size,
            show_progress=True
        )

        print(f"✅ {len(ids)}개 문서 Vector DB 적재 완료")

        # 통계 출력
        stats = self.vector_db.get_statistics()
        print("\n📊 Vector DB 통계:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return len(ids)

    def run_full_pipeline(
        self,
        urls: List[str] = None,
        max_pages: int = 30,
        reset_db: bool = False
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Args:
            urls: 수집할 URL 목록
            max_pages: 최대 수집 페이지 수
            reset_db: 기존 DB 초기화 여부

        Returns:
            실행 결과 통계
        """
        start_time = datetime.now()
        print("=" * 60)
        print("🚀 Vector DB 초기화 파이프라인 시작")
        print("=" * 60)

        try:
            # 1. 문서 수집
            documents = self.collect_documents(urls, max_pages)

            if not documents:
                print("❌ 수집된 문서가 없습니다.")
                return {"status": "failed", "error": "No documents collected"}

            # 2. 구조 기반 청킹
            chunked_docs = self.chunk_documents(documents)

            # 3. Vector DB 적재
            loaded_count = self.load_to_vector_db(
                chunked_docs,
                batch_size=100,
                reset=reset_db
            )

            # 4. 테스트 검색
            print("\n🔍 테스트 검색...")
            test_query = "LangChain에서 메모리를 사용하는 방법"
            results = self.vector_db.search(test_query, k=3)

            print(f"쿼리: '{test_query}'")
            print(f"검색 결과: {len(results)}개 문서")

            for i, doc in enumerate(results[:2], 1):
                print(f"\n[결과 {i}]")
                print(f"  내용: {doc.page_content[:200]}...")
                print(f"  메타데이터: {doc.metadata}")

            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()

            # 결과 반환
            result = {
                "status": "success",
                "documents_collected": len(documents),
                "chunks_created": len(chunked_docs),
                "documents_loaded": loaded_count,
                "execution_time": f"{execution_time:.2f}초",
                "vector_db_type": "Docker ChromaDB" if self.use_docker else "Local ChromaDB"
            }

            print("\n" + "=" * 60)
            print("✅ 파이프라인 완료!")
            print("=" * 60)

            for key, value in result.items():
                print(f"  {key}: {value}")

            return result

        except Exception as e:
            print(f"\n❌ 파이프라인 실행 실패: {e}")
            return {"status": "failed", "error": str(e)}


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Vector DB 초기화 및 데이터 로딩"
    )

    parser.add_argument(
        "--docker",
        action="store_true",
        help="Docker ChromaDB 사용 (기본: 로컬)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Docker ChromaDB 호스트 (기본: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Docker ChromaDB 포트 (기본: 8000)"
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
        default=30,
        help="최대 수집 페이지 수 (기본: 30)"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재생성"
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="테스트 모드 (5개 문서만)"
    )

    args = parser.parse_args()

    # 초기화
    initializer = VectorDBInitializer(
        use_docker=args.docker,
        docker_host=args.host,
        docker_port=args.port,
        collection_name=args.collection,
        embedding_model=args.embedding
    )

    # 테스트 모드
    if args.test_only:
        print("🧪 테스트 모드: 5개 문서만 처리")
        urls = LANGCHAIN_URLS[:5]
        max_pages = 5
    else:
        urls = LANGCHAIN_URLS
        max_pages = args.max_pages

    # 파이프라인 실행
    result = initializer.run_full_pipeline(
        urls=urls,
        max_pages=max_pages,
        reset_db=args.reset
    )

    # 결과 저장 (선택사항)
    if result["status"] == "success":
        # 로그 파일 저장
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"vector_db_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        with open(log_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n📝 로그 저장: {log_file}")


if __name__ == "__main__":
    main()