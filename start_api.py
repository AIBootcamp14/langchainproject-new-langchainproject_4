# start_api.py
import uvicorn
import os
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

# 2. Uvicorn 실행 설정
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
DEBUG = os.getenv("DEBUG", "True").lower() == "true"


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app", 
        host=API_HOST, 
        port=API_PORT, 
        reload=DEBUG,
        log_level=LOG_LEVEL
    )