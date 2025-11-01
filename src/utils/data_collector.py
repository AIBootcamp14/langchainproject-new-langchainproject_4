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
        수집할 샘플 URL 리스트 반환 (테스트용)
        """
        # 기존 10개의 샘플 URL 유지 (테스트용)
        urls: List[str] = [
            f"{self.base_url}docs/introduction",
            f"{self.base_url}docs/get_started/quickstart",
            f"{self.base_url}docs/concepts",
            f"{self.base_url}docs/modules/model_io/llms",
            f"{self.base_url}docs/modules/retrieval/vectorstores",
            f"{self.base_url}docs/modules/chains",
            f"{self.base_url}docs/modules/agents",
            f"{self.base_url}docs/modules/memory",
            f"{self.base_url}docs/expression_language",
            f"{self.base_url}docs/modules/callbacks",
        ]
        return urls

    def get_all_urls(self) -> List[str]:
        """
        수집할 모든 주요 LangChain 문서 URL 리스트 반환 (최종 적재용)

        NOTE: LangChain 문서 구조를 기반으로 주요 섹션 URL을 수동으로 정의함.
              전체 문서를 동적으로 찾으려면 별도 로직 (예: Recursive URL Loader)이 필요하나,
              여기서는 프로젝트 완료를 위해 주요 문서 목록을 확장함.
        """
        
        # 💡 [핵심 추가]: 전체 문서 적재를 위해 URL 목록을 대폭 확장
        all_urls: List[str] = [
            # 1. Getting Started
            f"{self.base_url}docs/introduction",
            f"{self.base_url}docs/get_started/quickstart",
            f"{self.base_url}docs/concepts",
            
            # 2. Key Modules
            f"{self.base_url}docs/modules/model_io/llms",
            f"{self.base_url}docs/modules/model_io/prompts",
            f"{self.base_url}docs/modules/model_io/chat",
            f"{self.base_url}docs/modules/retrieval/vectorstores",
            f"{self.base_url}docs/modules/retrieval/retriever",
            f"{self.base_url}docs/modules/chains",
            f"{self.base_url}docs/modules/agents",
            f"{self.base_url}docs/modules/agents/tools",
            f"{self.base_url}docs/modules/memory",
            
            # 3. Advanced Features (LCEL & Integrations)
            f"{self.base_url}docs/expression_language",
            f"{self.base_url}docs/integrations/llms/openai",
            f"{self.base_url}docs/integrations/llms/anthropic",
            f"{self.base_url}docs/integrations/vectorstores/chroma",
            f"{self.base_url}docs/integrations/vectorstores/faiss",
            f"{self.base_url}docs/modules/callbacks",
            
            # 4. Use Cases
            f"{self.base_url}docs/use_cases/question_answering",
            f"{self.base_url}docs/use_cases/summarization",
            f"{self.base_url}docs/use_cases/chatbots",
            
            # 5. Deployment/Ecosystem
            f"{self.base_url}docs/guides/deployment",
            f"{self.base_url}docs/guides/testing",
            f"{self.base_url}docs/guides/contributing",

            # 6. 추가적으로 팀이 다루기로 한 핵심 페이지가 있다면 여기에 추가
        ]
        return all_urls

    def extract_category(self, url: str) -> str:
        """URL에서 문서 카테고리 추출"""
        if "introduction" in url:
            return "introduction"
        elif "get_started" in url or "quickstart" in url:
            return "getting_started"
        elif "concepts" in url:
            return "concepts"
        # ... (이하 코드는 생략, 카테고리 로직은 그대로 유지)
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
        # ... (이하 코드는 변화 없음)
        """
        개별 페이지를 크롤링하여 LangChain Document 객체로 반환
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
        """
        if urls is None:
            # 💡 [핵심 수정]: URLs이 주어지지 않으면 전체 목록을 가져오도록 변경
            urls = self.get_all_urls() 
            
        # max_pages가 None일 경우 전체를 사용 (None인 경우 슬라이싱이 안 되므로 max_pages를 10000같은 큰 숫자로 대체)
        if max_pages is not None:
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
    # ... (테스트 코드는 그대로 유지)
    print("=" * 50)
    print("Data Collector 모듈 테스트")
    print("=" * 50)

    collector: DataCollector = DataCollector()

    # 1. 샘플 URL 확인
    print("\n1. 샘플 URL 확인 (10개):")
    sample_urls: List[str] = collector.get_sample_urls()
    print(f"  수집 대상 URL 개수: {len(sample_urls)}")

    # 2. 전체 URL 확인
    print("\n2. 전체 URL 확인:")
    all_urls: List[str] = collector.get_all_urls()
    print(f"  전체 URL 개수: {len(all_urls)}개")

    # 3. 실제 테스트 수집
    print("\n3. 테스트 수집 (2개 페이지, 지연 0.5초):")
    test_docs: List[Document] = collector.collect_documents(max_pages=2, delay=0.5)

    if test_docs:
        print(f"  수집된 문서: {len(test_docs)}개")
    else:
        print("  수집된 문서 없음.")

    print("\n테스트 완료!")