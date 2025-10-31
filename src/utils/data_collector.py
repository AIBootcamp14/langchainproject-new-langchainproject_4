# src/data_collector.py

"""
데이터 수집 모듈
LangChain 문서를 수집하여 Document 객체 리스트로 반환하는 기능
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# 써드파티 라이브러리
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader


class DataCollector:
    """LangChain 문서 수집 및 처리 클래스 (SQLite 기능 제거, 순수 크롤링 기능만 유지)"""

    def __init__(
        self,
        base_url: str = "https://python.langchain.com/",
    ) -> None:
        """
        DataCollector 초기화. (SQLite DB 경로는 제거됨)

        Args:
            base_url: LangChain 문서 기본 URL.
        """
        self.base_url = base_url

    def get_sample_urls(self) -> List[str]:
        """
        수집할 샘플 URL 리스트 반환 (initialize_vector_db.py의 목록과 일치시켜야 함)
        """
        urls: List[str] = [
            "https://python.langchain.com/docs/introduction",
            "https://python.langchain.com/docs/get_started/quickstart",
            "https://python.langchain.com/docs/concepts",
            "https://python.langchain.com/docs/modules/model_io/llms",
            "https://python.langchain.com/docs/modules/retrieval/vectorstores",
            "https://python.langchain.com/docs/modules/chains",
            "https://python.langchain.com/docs/modules/agents",
            "https://python.langchain.com/docs/modules/memory",
            "https://python.langchain.com/docs/expression_language",
            "https://python.langchain.com/docs/modules/callbacks",
        ]
        return urls

    def extract_category(self, url: str) -> str:
        """URL에서 문서 카테고리 추출"""
        if "introduction" in url:
            return "introduction"
        elif "get_started" in url or "quickstart" in url:
            return "getting_started"
        elif "concepts" in url:
            return "concepts"
        elif "modules/model_io" in url:
            return "model_io"
        elif "modules/retrieval" in url:
            return "retrieval"
        elif "modules/chains" in url:
            return "chains"
        elif "modules/agents" in url:
            return "agents"
        elif "modules/memory" in url:
            return "memory"
        elif "expression_language" in url or "lcel" in url:
            return "lcel"
        else:
            return "general"

    def crawl_page(self, url: str) -> Optional[Document]:
        """
        개별 페이지를 크롤링하여 LangChain Document 객체로 반환

        Args:
            url: 크롤링할 URL.

        Returns:
            LangChain Document 객체 또는 크롤링 실패 시 None.
        """
        try:
            # WebBaseLoader를 사용해 페이지 로드
            loader = WebBaseLoader(url)
            docs: List[Document] = loader.load()

            if not docs:
                return None

            doc: Document = docs[0]

            # URL에서 카테고리 추출
            category: str = self.extract_category(url)

            # 문서 ID 생성
            doc_id: str = url.replace(self.base_url, "").replace("/", "_").replace(".html", "")
            
            # 메타데이터 정리 및 추가
            metadata: Dict[str, Any] = doc.metadata
            metadata["doc_id"] = doc_id
            metadata["url"] = url
            metadata["category"] = category
            metadata["timestamp"] = datetime.now().isoformat()
            
            # title이 없으면 URL을 사용
            if "title" not in metadata or not metadata["title"]:
                metadata["title"] = url.split('/')[-1]

            return Document(page_content=doc.page_content, metadata=metadata)

        except Exception as e:
            # tqdm 때문에 출력 방지하고 대신 로그 파일에 기록하거나, 에러 카운트만 하는 것이 좋음
            print(f"\n페이지 크롤링 실패 ({url}): {e}")
            return None

    def collect_documents(
        self,
        urls: Optional[List[str]] = None,
        max_pages: int = 100,
        delay: float = 1.0,
    ) -> List[Document]:
        """
        문서 수집 메인 함수

        Args:
            urls: 수집할 URL 리스트 (None이면 샘플 URL 사용).
            max_pages: 최대 수집 페이지 수.
            delay: 요청 간 대기 시간 (초).

        Returns:
            수집된 Document 리스트.
        """
        if urls is None:
            urls = self.get_sample_urls()

        urls = urls[:max_pages]
        documents: List[Document] = []
        
        print(f"총 {len(urls)}개 페이지 수집 시작...")

        for url in tqdm(urls, desc="크롤링 진행"):
            # 페이지 크롤링
            doc: Optional[Document] = self.crawl_page(url)

            if doc:
                documents.append(doc)
                
            # 대기 시간
            time.sleep(delay)

        print(f"총 {len(documents)}개 문서 수집 완료")
        return documents


if __name__ == "__main__":
    # 모듈 테스트
    print("=" * 50)
    print("Data Collector 모듈 테스트")
    print("=" * 50)

    collector: DataCollector = DataCollector()

    # 1. 샘플 URL 확인
    print("\n1. 샘플 URL 확인:")
    sample_urls: List[str] = collector.get_sample_urls()
    print(f"  수집 대상 URL 개수: {len(sample_urls)}")
    print(f"  첫 번째 URL: {sample_urls[0]}")

    # 2. 실제 테스트 수집
    print("\n2. 테스트 수집 (2개 페이지, 지연 0.5초):")
    test_docs: List[Document] = collector.collect_documents(max_pages=2, delay=0.5)

    if test_docs:
        print(f"  수집된 문서: {len(test_docs)}개")
        print(f"  첫 번째 문서 제목: {test_docs[0].metadata.get('title')}")
        print(f"  첫 번째 문서 카테고리: {test_docs[0].metadata.get('category')}")
        print(f"  첫 번째 문서 내용 일부: {test_docs[0].page_content[:100]}...")
    else:
        print("  수집된 문서 없음.")

    print("\n테스트 완료!")