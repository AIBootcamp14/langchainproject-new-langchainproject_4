# start_api.py

# 이 파일은 이제 사용되지 않습니다.
# 실행 명령어는 docker-compose.yml의 command 필드에서 uvicorn을 직접 호출하도록 변경되었습니다.
# import os
# import uvicorn
# from dotenv import load_dotenv

# load_dotenv()

# HOST: str = os.getenv("HOST", "0.0.0.0")
# PORT: int = int(os.getenv("PORT", 8000))

# if __name__ == "__main__":
#     print(f"🚀 FastAPI 서버 시작 중: http://{HOST}:{PORT}")
    
#     # reload=True 옵션이 Docker 환경에서 문제를 일으켜서 직접 실행 방식으로 변경됨.
#     uvicorn.run(
#         "src.main:app", 
#         host=HOST, 
#         port=PORT, 
#         reload=False # 로컬에서 실행할 경우를 대비해 False로 변경
#     )