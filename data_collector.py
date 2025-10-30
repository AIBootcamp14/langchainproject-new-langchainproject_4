"""데이터 수집 모듈
LangChain 문서를 수집하고 처리하는 기능"""

import os
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from vector_database import VectorDatabase
from advanced_text_splitter import StructuredTextSplitter, HTMLStructuredSplitter


class LangChainDataCollector:
    """LangChain 문서 수집 및 처리 클래스"""

    def __init__(
        self,
        base_url: str = "https://python.langchain.com/",
        db_path: str = "./data/langchain.db"
    ):
        """
        DataCollector 초기화

        Args:
            base_url: LangChain 문서 기본 URL
            db_path: SQLite 데이터베이스 경로
        """
        self.base_url = base_url
        self.db_path = db_path

        # 데이터 저장 디렉토리 생성
        Path("./data/raw").mkdir(parents=True, exist_ok=True)
        Path("./data/processed").mkdir(parents=True, exist_ok=True)

        # SQLite DB 초기화
        self.init_database()

    def init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # documents 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id VARCHAR(255) UNIQUE NOT NULL,
                title VARCHAR(500),
                url TEXT,
                category VARCHAR(100),
                module_name VARCHAR(200),
                content TEXT NOT NULL,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # code_examples 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id VARCHAR(255),
                language VARCHAR(50),
                code TEXT NOT NULL,
                description TEXT,
                imports TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        # api_references 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name VARCHAR(200),
                method_name VARCHAR(200),
                parameters TEXT,
                return_type VARCHAR(100),
                description TEXT,
                doc_id VARCHAR(255),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)

        conn.commit()
        conn.close()
        print(f"데이터베이스 초기화 완료: {self.db_path}")

    def get_sample_urls(self) -> List[str]:
        """
        수집할 샘플 URL 리스트 반환
        실제 크롤링 구현시에는 사이트맵이나 링크를 따라가며 URL을 수집해야 하지만,
        데모용으로 주요 문서 페이지 URL을 반환합니다
        """
        urls = [
            "https://python.langchain.com/docs/introduction",
            "https://python.langchain.com/docs/get_started/quickstart",
            "https://python.langchain.com/docs/concepts",
            "https://python.langchain.com/docs/tutorials",
            "https://python.langchain.com/docs/how_to",
            "https://python.langchain.com/docs/expression_language",
            "https://python.langchain.com/docs/modules/model_io/llms",
            "https://python.langchain.com/docs/modules/model_io/prompts",
            "https://python.langchain.com/docs/modules/retrieval/document_loaders",
            "https://python.langchain.com/docs/modules/retrieval/text_splitters",
            "https://python.langchain.com/docs/modules/retrieval/vectorstores",
            "https://python.langchain.com/docs/modules/chains",
            "https://python.langchain.com/docs/modules/agents",
            "https://python.langchain.com/docs/modules/memory",
            "https://python.langchain.com/docs/modules/callbacks"
        ]
        return urls

    def crawl_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        개별 페이지 크롤링

        Args:
            url: 크롤링할 URL

        Returns:
            페이지 데이터 딕셔너리
        """
        try:
            # 웹페이지 로드
            loader = WebBaseLoader(url)
            docs = loader.load()

            if not docs:
                return None

            doc = docs[0]

            # URL에서 카테고리 추출
            category = self.extract_category(url)

            # 문서 ID 생성
            doc_id = url.replace(self.base_url, "").replace("/", "_")

            page_data = {
                "doc_id": doc_id,
                "title": doc.metadata.get("title", "Untitled"),
                "url": url,
                "category": category,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": datetime.now().isoformat()
            }

            return page_data

        except Exception as e:
            print(f"페이지 크롤링 실패 ({url}): {e}")
            return None

    def extract_category(self, url: str) -> str:
        """URL에서 카테고리 추출"""
        if "introduction" in url:
            return "introduction"
        elif "get_started" in url or "quickstart" in url:
            return "getting_started"
        elif "concepts" in url:
            return "concepts"
        elif "tutorials" in url:
            return "tutorials"
        elif "how_to" in url:
            return "how_to"
        elif "expression_language" in url or "lcel" in url:
            return "lcel"
        elif "modules" in url:
            if "model_io" in url:
                return "model_io"
            elif "retrieval" in url:
                return "retrieval"
            elif "chains" in url:
                return "chains"
            elif "agents" in url:
                return "agents"
            elif "memory" in url:
                return "memory"
            else:
                return "modules"
        elif "api" in url:
            return "api_reference"
        else:
            return "general"

    def extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """
        컨텐츠에서 코드 예제 추출

        Args:
            content: 페이지 컨텐츠 내용

        Returns:
            코드 예제 리스트
        """
        soup = BeautifulSoup(content, 'html.parser')
        code_blocks = soup.find_all(['code', 'pre'])

        examples = []
        for block in code_blocks:
            code_text = block.get_text().strip()
            if len(code_text) > 50:  # 의미있는 코드만 저장
                # import 문 추출
                imports = []
                for line in code_text.split('\n'):
                    if line.strip().startswith(('import ', 'from ')):
                        imports.append(line.strip())

                examples.append({
                    "code": code_text,
                    "language": "python",
                    "imports": "\n".join(imports) if imports else None
                })

        return examples

    def save_to_database(self, page_data: Dict[str, Any]):
        """
        데이터를 SQLite 데이터베이스에 저장

        Args:
            page_data: 페이지 데이터
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # documents 테이블에 저장
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (doc_id, title, url, category, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                page_data["doc_id"],
                page_data["title"],
                page_data["url"],
                page_data["category"],
                page_data["content"],
                page_data["timestamp"],
                page_data["timestamp"]
            ))

            # 코드 예제 저장
            code_examples = self.extract_code_examples(page_data["content"])
            for example in code_examples:
                cursor.execute("""
                    INSERT INTO code_examples
                    (doc_id, language, code, imports)
                    VALUES (?, ?, ?, ?)
                """, (
                    page_data["doc_id"],
                    example["language"],
                    example["code"],
                    example["imports"]
                ))

            conn.commit()

        except Exception as e:
            print(f"데이터베이스 저장 실패: {e}")
            conn.rollback()
        finally:
            conn.close()

    def collect_documents(
        self,
        urls: Optional[List[str]] = None,
        max_pages: int = 100,
        delay: float = 1.0
    ) -> List[Document]:
        """
        문서 수집 메인 함수

        Args:
            urls: 수집할 URL 리스트 (None이면 샘플 URL 사용)
            max_pages: 최대 수집 페이지 수
            delay: 요청 간 대기 시간 (초)

        Returns:
            수집된 Document 리스트
        """
        if urls is None:
            urls = self.get_sample_urls()

        urls = urls[:max_pages]
        documents = []

        print(f"총 {len(urls)}개 페이지 수집 시작...")

        for url in tqdm(urls, desc="크롤링 진행"):
            # 페이지 크롤링
            page_data = self.crawl_page(url)

            if page_data:
                # DB에 저장
                self.save_to_database(page_data)

                # Document 객체 생성
                doc = Document(
                    page_content=page_data["content"],
                    metadata={
                        "doc_id": page_data["doc_id"],
                        "title": page_data["title"],
                        "url": page_data["url"],
                        "category": page_data["category"],
                        "timestamp": page_data["timestamp"]
                    }
                )
                documents.append(doc)

                # JSON 파일로도 저장
                json_path = f"./data/raw/{page_data['doc_id']}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)

            # 대기 시간
            time.sleep(delay)

        print(f"총 {len(documents)}개 문서 수집 완료")
        return documents

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_structured_splitter: bool = True
    ) -> List[Document]:
        """
        문서를 청크로 분할 (구조 기반 또는 일반 분할)

        Args:
            documents: 원본 문서 리스트
            chunk_size: 청크당 최대 길이
            chunk_overlap: 중복 길이
            use_structured_splitter: 구조 기반 분할기 사용 여부

        Returns:
            청크 분할된 문서 리스트
        """
        if use_structured_splitter:
            # 구조 기반 텍스트 분할기 사용
            print("구조 기반 텍스트 분할기 사용 중...")
            text_splitter = StructuredTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                code_block_max_size=2000,  # 코드 블록 최대 크기
                preserve_code_blocks=True,
                preserve_functions=True,
                preserve_markdown_structure=True
            )
        else:
            # 기존 RecursiveCharacterTextSplitter 사용 (fallback)
            print("일반 텍스트 분할기 사용 중...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

        chunked_docs = []
        code_block_count = 0
        function_count = 0

        for doc in documents:
            if use_structured_splitter:
                # 구조 기반 분할
                chunks = text_splitter.split_documents([doc])

                # 통계 수집
                for chunk in chunks:
                    if chunk.metadata.get('chunk_type') == 'code':
                        code_block_count += 1
                    if chunk.metadata.get('functions'):
                        function_count += len(chunk.metadata['functions'])
            else:
                # 일반 분할
                chunks = text_splitter.split_documents([doc])

            # 각 청크에 원본 문서 정보 유지
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                chunk.metadata["original_doc_id"] = doc.metadata.get("doc_id", "unknown")
                chunked_docs.append(chunk)

        print(f"총 {len(documents)}개 문서를 {len(chunked_docs)}개 청크로 분할")

        if use_structured_splitter:
            print(f"  - 코드 블록 청크: {code_block_count}개")
            print(f"  - 함수 정의 보존: {function_count}개")

        return chunked_docs

    def process_and_store(
        self,
        urls: Optional[List[str]] = None,
        max_pages: int = 100
    ) -> bool:
        """
        데이터 수집, 처리, 저장을 한 번에 수행

        Args:
            urls: 수집할 URL 리스트
            max_pages: 최대 수집 페이지 수

        Returns:
            성공 여부
        """
        try:
            # 1. 문서 수집
            documents = self.collect_documents(urls, max_pages)

            if not documents:
                print("수집된 문서가 없습니다.")
                return False

            # 2. 문서 청크 분할
            chunked_docs = self.chunk_documents(documents)

            # 3. 벡터 DB에 저장
            from vector_database import VectorDatabase
            vdb = VectorDatabase()
            vdb.init_vectorstore()
            vdb.add_documents(chunked_docs)

            print(f"전체 처리 완료: {len(chunked_docs)}개 청크 저장됨")
            return True

        except Exception as e:
            print(f"처리 중 오류 발생: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보 반환"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 전체 문서 수
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["total_documents"] = cursor.fetchone()[0]

        # 카테고리별 문서 수
        cursor.execute("SELECT category, COUNT(*) FROM documents GROUP BY category")
        stats["documents_by_category"] = dict(cursor.fetchall())

        # 코드 예제 수
        cursor.execute("SELECT COUNT(*) FROM code_examples")
        stats["total_code_examples"] = cursor.fetchone()[0]

        conn.close()
        return stats


# 간단한 사용 함수들
def crawl_langchain_docs(max_pages: int = 10) -> List[Document]:
    """
    LangChain 문서를 크롤링합니다.

    Args:
        max_pages: 최대 수집 페이지 수

    Returns:
        수집된 Document 리스트
    """
    collector = LangChainDataCollector()
    return collector.collect_documents(max_pages=max_pages)


def process_documents(documents: List[Document]) -> List[Document]:
    """
    문서를 처리합니다 (청크 분할).

    Args:
        documents: 원본 문서 리스트

    Returns:
        처리된 문서 리스트
    """
    collector = LangChainDataCollector()
    return collector.chunk_documents(documents)


if __name__ == "__main__":
    # 모듈 테스트
    print("=" * 50)
    print("Data Collector 모듈 테스트\n")

    # 데이터 수집기 초기화
    collector = LangChainDataCollector()

    # 1. 샘플 URL 확인
    print("1단계: 샘플 URL 확인")
    sample_urls = collector.get_sample_urls()
    print(f"  수집 대상 URL 개수: {len(sample_urls)}")
    print(f"  첫 번째 URL: {sample_urls[0]}")

    # 2. 실제 테스트 수집
    print("\n2단계: 테스트 수집 (3개 페이지)")
    test_docs = collector.collect_documents(max_pages=3)

    if test_docs:
        print(f"  수집된 문서: {len(test_docs)}개")
        print(f"  첫 번째 문서 제목: {test_docs[0].metadata.get('title')}")

    # 3. 통계 확인
    print("\n3단계: 데이터베이스 통계")
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n테스트 완료!")
