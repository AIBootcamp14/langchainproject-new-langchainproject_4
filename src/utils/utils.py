# src/utils/utils.py

"""
공통적으로 사용되는 유틸리티 함수 모음
"""

import os
import re
import hashlib
from typing import List, Optional, Any, Tuple


def ensure_directory(path: str) -> None:
    """
    주어진 경로에 디렉터리가 없으면 생성한다.
    
    Args:
        path: 생성할 디렉터리 경로
    """
    # 💡 [핵심 수정]: exist_ok=True 명시 (PEP 20: 명시적인 것이 암시적인 것보다 낫다)
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    """
    텍스트에서 불필요한 공백, 개행 문자 등을 제거하여 정제한다.
    
    Args:
        text: 정제할 문자열
        
    Returns:
        정제된 문자열
    """
    # 여러 개의 개행 문자를 공백 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    # 문자열 앞뒤 공백 제거
    return text.strip()


def generate_document_hash(doc_content: str, doc_metadata: Optional[Any] = None) -> str:
    """
    문서 내용과 메타데이터(선택 사항)를 기반으로 고유 해시 값을 생성한다.
    
    Args:
        doc_content: 문서의 내용 (str)
        doc_metadata: 문서의 메타데이터 (Dict, str 등)
        
    Returns:
        문서의 SHA256 해시값 (str)
    """
    # 메타데이터가 있으면 문자열로 변환하여 내용에 추가
    combined_data: str = doc_content
    if doc_metadata is not None:
        # dict 형태일 수 있으므로 안전하게 문자열로 변환
        combined_data += str(doc_metadata) 

    # SHA256 해시 생성
    # PEP 8: 변수 이름은 소문자와 밑줄로 (snake_case)
    sha256_hash: str = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
    
    return sha256_hash


if __name__ == "__main__":
    
    # 1. ensure_directory 테스트
    test_dir: str = "test_temp_dir/sub_dir"
    print(f"1. 디렉터리 생성 테스트: '{test_dir}'")
    ensure_directory(test_dir)
    if os.path.isdir(test_dir):
        print("   ✅ 생성 성공")
    else:
        print("   ❌ 생성 실패")
        
    # 2. clean_text 테스트
    raw_text: str = "  안녕하세요.\n\n파이썬    코드를  테스트합니다.  "
    cleaned_text: str = clean_text(raw_text)
    print(f"\n2. 텍스트 정제 테스트:")
    print(f"   원본: '{raw_text}'")
    print(f"   결과: '{cleaned_text}'")
    if "  " not in cleaned_text and cleaned_text.startswith("안녕하세요"):
        print("   ✅ 정제 성공")
    else:
        print("   ❌ 정제 실패")

    # 3. generate_document_hash 테스트
    content_a: str = "테스트 문서 내용 A"
    content_b: str = "테스트 문서 내용 B"
    hash_a1: str = generate_document_hash(content_a)
    hash_a2: str = generate_document_hash(content_a)
    hash_b: str = generate_document_hash(content_b)
    
    print("\n3. 해시 생성 테스트:")
    print(f"   Hash A1: {hash_a1}")
    print(f"   Hash A2: {hash_a2}")
    print(f"   Hash B:  {hash_b}")
    
    if hash_a1 == hash_a2 and hash_a1 != hash_b:
        print("   ✅ 해시 일관성 및 고유성 성공")
    else:
        print("   ❌ 해시 테스트 실패")
        
    # 테스트 후 생성된 디렉터리 삭제 (PEP 20: 삭제도 명시적으로)
    if os.path.isdir("test_temp_dir"):
        import shutil
        shutil.rmtree("test_temp_dir")
        print(f"\n테스트 디렉터리 '{'test_temp_dir'}' 삭제 완료.")