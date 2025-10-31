# src/utils/utils.py

"""
RAG 프로젝트 공통 유틸리티 함수 모음
- 디렉토리 관리, 텍스트 처리, 문서 ID 생성 등
"""

import os
import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 로거 설정
logger = logging.getLogger(__name__)


# --- 파일 시스템 유틸리티 ---
def ensure_directory(path: str) -> Path:
    """
    디렉토리가 존재하지 않으면 생성합니다.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    JSON 파일을 읽어 딕셔너리로 반환합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 오류: {e}")
        return {}


def write_json_file(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    딕셔너리 데이터를 JSON 파일로 저장합니다.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"파일 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")


# --- 텍스트 및 ID 처리 유틸리티 ---
def clean_text(text: str) -> str:
    """
    텍스트 정제 (공백 정리, 제어 문자 제거).
    """
    # 여러 개의 공백을 하나의 공백으로 대체
    text = re.sub(r'\s+', ' ', text)
    
    # 텍스트 양 끝 공백 제거
    text = text.strip()

    # 제어 문자 처리 (Zero-width space, BOM 등)
    text = text.replace('\u200b', '') # Zero-width space
    text = text.replace('\ufeff', '') # BOM

    return text


def generate_hash(text: str) -> str:
    """
    텍스트의 SHA256 해시를 생성합니다.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def generate_document_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    문서의 고유 ID를 생성합니다. 
    URL이 있다면 URL과 내용의 일부를 합쳐 ID를 만듭니다.
    """
    id_source = content

    if metadata and 'source' in metadata and isinstance(metadata['source'], str):
        # LangChain의 기본 메타데이터 키는 'source'를 사용함.
        id_source = f"{metadata['source']}_{content[:100]}"
    elif metadata and 'title' in metadata:
        id_source = f"{metadata['title']}_{content[:100]}"

    # ID는 해시값의 앞 16자리만 사용 (충분히 고유함)
    return generate_hash(id_source)[:16]


# --- 환경 설정 유틸리티 ---
def load_env_file(env_path: str = ".env") -> None:
    """
    .env 파일 환경 변수를 로드합니다.
    """
    try:
        load_dotenv(env_path)
        logger.info(f"환경 변수 파일 로드: {env_path}")
    except Exception as e:
        logger.error(f"환경 변수 파일 로드 실패: {e}")


if __name__ == "__main__":
    # 테스트 스크립트
    logging.basicConfig(level=logging.INFO)
    print(">>> 유틸리티 모듈 테스트 시작")

    # 1. 텍스트 처리 및 ID 생성 테스트
    test_text = "This is a test <b>text</b> with   \n\n\u200b spaces."
    print(f"원본: '{test_text}'")
    cleaned_text = clean_text(test_text)
    print(f"정제: '{cleaned_text}'")

    doc_metadata = {"source": "http://example.com/doc1", "title": "Test Document"}
    doc_id = generate_document_id(cleaned_text, doc_metadata)
    print(f"문서 ID: {doc_id}")

    # 2. 파일 시스템 테스트 (생략)
    print(">>> 유틸리티 모듈 테스트 완료")