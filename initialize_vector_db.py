# initialize_vector_db.py

"""
Vector DB 초기화 스크립트 (최종)
LangChain 문서를 수집하고 구조 기반 청킹 후 ChromaDB Vector DB에 적재합니다.
"""

import os
import sys
from pathlib import Path
import time


# 🌟 경로 추가 및 디버깅 코드 시작 🌟
current_dir = str(Path(__file__).parent)
sys.path.append(current_dir) # <--- 이 부분이 경로를 추가하는 핵심이야.

print("="*30)
print(f"현재 스크립트 경로: {current_dir}")

is_src_exist = Path(current_dir, "src").is_dir()
print(f"폴더 안에 'src' 폴더 존재 여부: {is_src_exist}")

print("sys.path에 추가된 경로들:")
for p in sys.path:
    if "lang" in p: # 프로젝트 폴더 이름으로 필터링
        print(f"  -> {p}")
print("="*30)
# 🌟 경로 추가 및 디버깅 코드 끝 🌟

from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document

# 로컬 모듈 임포트
# 팀원 4와 팀원 1이 완성한 실제 모듈 사용
from src.utils.data_collector import DataCollector
from src.utils.chunking_strategy import StructuredTextSplitter
from src.modules.vector_database import VectorDatabaseClient

# 환경 변수 로드 (API 키 및 DB 설정)
load_dotenv() 

# --- 설정 값 ---
# 환경 변수가 없으면 기본값 사용
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "langchain_docs")
MAX_PAGES_TO_CRAWL: int = 10 # 전체 문서를 적재하기 전에 테스트용으로 10개로 제한
CRAWL_DELAY_SECONDS: float = 1.0
RESET_DB: bool = True # DB를 새로 만들지 여부 (테스트 시 True 권장)


def run_data_ingestion() -> None:
    """전체 데이터 적재 파이프라인 실행"""
    start_time = time.time()
    
    print("=" * 60)
    print(f"🚀 RAG 데이터 적재 파이프라인 시작 (컬렉션: {COLLECTION_NAME})")
    print("=" * 60)
    
    # 1. 문서 수집 (DataCollector 사용)
    print("\n[단계 1/4] LangChain 문서 수집 시작...")
    collector = DataCollector()
    
    # DataCollector 내부의 get_sample_urls 사용
    urls_to_crawl = collector.get_sample_urls() 
    
    raw_documents: List[Document] = collector.collect_documents(
        urls=urls_to_crawl,
        max_pages=MAX_PAGES_TO_CRAWL,
        delay=CRAWL_DELAY_SECONDS,
    )
    
    if not raw_documents:
        print("🛑 수집된 문서가 없습니다. 크롤링 URL 또는 웹 서버 상태를 점검하세요.")
        return
        
    print(f"✅ 문서 수집 완료: 총 {len(raw_documents)}개 문서")
    
    
    # 2. 구조 기반 텍스트 분할 (StructuredTextSplitter 사용)
    print("\n[단계 2/4] 구조 기반 텍스트 분할 및 청킹 시작...")
    # 팀원 1의 고급 분할기 사용 (코드 블록 보존)
    text_splitter = StructuredTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        preserve_code_blocks=True,
    )
    
    chunks: List[Document] = text_splitter.split_documents(raw_documents)
    
    if not chunks:
        print("🛑 분할된 청크가 없습니다. 분할 로직을 점검하세요.")
        return
        
    print(f"✅ 문서 분할 완료: 총 {len(chunks)}개 청크 생성")


    # 3. 벡터 데이터베이스 클라이언트 초기화 및 연결 확인
    print("\n[단계 3/4] ChromaDB 연결 및 임베딩 모델 초기화...")
    vdb_client = VectorDatabaseClient(collection_name=COLLECTION_NAME)
    
    if not vdb_client.health_check():
        print("🛑 ChromaDB 서버 연결 실패. Docker Compose가 실행 중인지 확인하세요.")
        return
    
    # reset 설정에 따라 기존 데이터 삭제 후 초기화
    vectorstore = vdb_client.init_vectorstore(reset=RESET_DB)
    print(f"✅ 벡터 저장소 초기화 완료 (컬렉션: {COLLECTION_NAME})")


    # 4. 벡터 저장소에 청크 적재
    print(f"\n[단계 4/4] {len(chunks)}개 청크를 벡터 저장소에 적재 시작...")
    
    try:
        # ChromaDB에 Document 리스트를 직접 추가 (자동으로 임베딩 및 저장 수행)
        vectorstore.add_documents(documents=chunks)
        
        # 적재 후 최종 문서 수 확인
        final_count = vectorstore._collection.count()
        print(f"🎉 모든 청크 적재 완료! (총 {final_count}개 청크)")
        
    except Exception as e:
        print(f"❌ 데이터 적재 중 치명적인 오류 발생: {e}")
        
    execution_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 파이프라인 완료! 총 실행 시간: {execution_time:.2f}초")
    print("=" * 60)


if __name__ == "__main__":
    run_data_ingestion()