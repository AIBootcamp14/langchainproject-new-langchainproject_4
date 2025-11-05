"""
전체 데이터 수집 및 벡터 데이터베이스 초기화 스크립트.
(ChromaDB 연결 포트 및 환경 변수 수정)
"""

import os
import argparse 
import time
from typing import List, Dict, Any, Optional, Final

# 써드파티 라이브러리
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma

# 프로젝트 모듈
from src.utils.data_collector import DataCollector
from src.utils.utils import ensure_directory, generate_document_hash
from src.utils.chunking_strategy import CodeBlockPreservingSplitter
from src.modules.vector_database import VectorDatabaseClient # 이 모듈을 수정해야 함
from src.modules.llm import get_embeddings


# --- 설정 및 상수 ---
SOURCE_DATA_DIR: Final[str] = "data/source_documents"
EMBEDDING_MODEL_NAME: Final[str] = "solar-embedding-1-large"
COLLECTION_NAME: Final[str] = "langchain_docs"

# --------------------


def initialize_db(
    documents: List[Document], 
    reset_db: bool = False
) -> None:
    """
    수집된 문서를 청킹하고 벡터 데이터베이스에 적재한다.
    """
    
    print("=" * 60)
    print(f"2. 벡터 데이터베이스 초기화 및 적재 시작 (Reset: {reset_db})")
    print("=" * 60)
    
    # 💡 [핵심 수정]: 데이터 로드 스크립트 실행 시 ChromaDB 연결 설정을 오버라이드
    # Docker-compose의 8001:8000 매핑을 사용 (FastAPI 8000과 충돌 방지)
    os.environ["CHROMA_HOST"] = "localhost"
    os.environ["CHROMA_PORT"] = "8001"
    
    # 1. DB 클라이언트 초기화 및 연결
    vdb_client: VectorDatabaseClient = VectorDatabaseClient(
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL_NAME
    )
    
    # ChromaDB 연결 확인
    if not vdb_client.health_check():
        raise ConnectionError("ChromaDB 서버에 연결할 수 없습니다. 스크립트를 중단합니다.")

    # 벡터 저장소 초기화 (reset 인자 전달)
    vectorstore: Chroma = vdb_client.init_vectorstore(reset=reset_db)

    if not documents:
        print("경고: 적재할 문서가 없습니다. DB 초기화만 완료되었습니다.")
        return

    # 2. 문서 분할 (Chunking)
    print(f"총 {len(documents)}개 문서 분할 시작 (Custom Splitter 사용)")

    # CodeBlockPreservingSplitter 사용
    text_splitter: RecursiveCharacterTextSplitter = CodeBlockPreservingSplitter(
        chunk_size=1500, 
        chunk_overlap=200, 
    )

    chunks: List[Document] = text_splitter.split_documents(documents)
    print(f"✅ 문서 분할 완료. 총 {len(chunks)}개 청크 생성됨.")

    # 3. 문서 해시 생성 및 메타데이터 추가
    for chunk in chunks:
        # 청크 레벨에서 고유 해시 생성
        chunk.metadata["chunk_hash"] = generate_document_hash(
            chunk.page_content, 
            chunk.metadata.get("url")
        )

    # 4. 벡터 DB에 청크 적재
    print(f"총 {len(chunks)}개 청크를 벡터 DB에 적재 중...")
    
    start_time = time.time()
    ids: List[str] = vectorstore.add_documents(chunks)
    end_time = time.time()
    
    print(f"✅ 적재 완료! (총 {len(ids)}개 문서 적재, 소요 시간: {end_time - start_time:.2f}초)")


def parse_arguments() -> argparse.Namespace:
    """명령줄 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="LangChain 문서를 크롤링하고 벡터 데이터베이스(ChromaDB)를 초기화합니다."
    )
    parser.add_argument(
        "--reset",
        action="store_true", 
        help="기존 ChromaDB 컬렉션을 삭제하고 새로 만듭니다."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="크롤링할 최대 페이지 수 (개발/테스트 용). 0 또는 None이면 전체 크롤링."
    )
    
    return parser.parse_args()


def main():
    """스크립트 메인 실행 함수"""
    load_dotenv()
    
    args: argparse.Namespace = parse_arguments()
    reset_db: bool = args.reset
    max_pages: Optional[int] = args.max_pages if args.max_pages > 0 else None 
    
    print("=" * 60)
    print("1. 데이터 수집 시작")
    print(f"  - DB 초기화 여부 (--reset): {reset_db}")
    print(f"  - 최대 페이지 수 (--max-pages): {max_pages if max_pages is not None else '전체'}")
    print("=" * 60)
    
    try:
        collector: DataCollector = DataCollector()
        documents: List[Document] = collector.collect_documents(
            max_pages=max_pages, 
            delay=0.5
        )
        
        if not documents:
            print("경고: 수집된 문서가 없어 적재 단계를 건너뜁니다.")
        
        initialize_db(documents=documents, reset_db=reset_db)

    except ConnectionError as e:
        print(f"\n❌ [치명적 오류 - 연결 실패]: {e}")
        print("ChromaDB 서버(Docker 컨테이너)가 실행 중인지 확인해주세요.")
    except ValueError as e:
        print(f"\n❌ [치명적 오류 - 설정 실패]: {e}")
    except Exception as e:
        print(f"\n❌ [예상치 못한 오류]: {e.__class__.__name__} - {e}")
    finally:
        print("\n=== 데이터베이스 초기화 및 적재 스크립트 종료 ===")


if __name__ == "__main__":
    main()