"""
FastAPI 서버 실행을 위한 엔트리 포인트 스크립트.
"""
import os
# --- [긴급 디버깅: vector_database.py 파일 버전 확인] ---
# Docker가 이전 캐시를 사용해 파일을 복사하지 않는지 확인하기 위한 코드
import inspect
from src.modules.vector_database import VectorDatabaseClient

# health_check 함수의 코드를 문자열로 추출
# 이 부분이 에러 없이 실행된다는 것은 import는 성공했다는 의미
try:
    health_check_source = inspect.getsource(VectorDatabaseClient.health_check)
    
    # 우리가 수정한 핵심 문자열(tenant 명시)이 코드 안에 있는지 확인
    if 'tenant="default_tenant"' not in health_check_source:
        print("FATAL_CODE_ERROR: 'src/modules/vector_database.py' 파일이 최신 버전으로 복사되지 않았습니다! (tenant 명시 누락)")
        # 강제 종료하여 문제를 명확히 함
        os._exit(1)
    else:
        print("DEBUG_CODE_CHECK: vector_database.py 최신 코드 (tenant 명시) 확인 완료.")
except Exception as e:
    # inspect 실패 시 (예: 파일이 예상과 다르게 로드됨)
    print(f"FATAL_CODE_ERROR: 코드 검증 실패 - {e}")
    os._exit(1)
# --- [디버깅 끝] ---

import uvicorn
from dotenv import load_dotenv

# 환경 변수를 로드
load_dotenv()

# 서버 실행 설정
# Uvicorn의 호스트와 포트는 Docker Compose에서 설정되지만, 로컬 실행을 위해 기본값을 정의
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    print(f"🚀 FastAPI 서버 시작 중: http://{HOST}:{PORT}")
    
    # src.main 모듈의 app 객체를 Uvicorn으로 실행
    uvicorn.run(
        "src.main:app", 
        host=HOST, 
        port=PORT, 
        reload=True # 개발 환경에서는 reload=True로 설정
    )