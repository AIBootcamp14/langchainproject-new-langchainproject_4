# start_api.py

"""
FastAPI 서버 실행을 위한 엔트리 포인트 스크립트.
"""
import os
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