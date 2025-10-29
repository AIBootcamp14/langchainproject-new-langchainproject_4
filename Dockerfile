# Dockerfile

# --- 1단계: 빌드 환경 설정 ---
# 안정적인 Python 3.11.9 버전을 사용
FROM python:3.11.9-slim-bookworm as builder

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
# pip 캐시 사용 안함 (--no-cache-dir) 으로 이미지 크기 최소화
RUN pip install --no-cache-dir -r requirements.txt

# --- 2단계: 최종 실행 환경 구성 ---
FROM python:3.11.9-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 경로 설정: src/ 디렉토리를 모듈 경로에 포함시켜 절대 경로로 임포트 가능하게 함
ENV PYTHONPATH /app
# 출력 버퍼링 비활성화 (로그 실시간 확인에 유용)
ENV PYTHONUNBUFFERED 1

# 빌드 단계에서 설치된 라이브러리 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 실행 파일(uvicorn 포함)이 있는 /usr/local/bin 디렉토리를 복사
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 코드 복사 (requirements.txt는 이미 설치됨)
COPY src/ src/

# FastAPI의 기본 포트 8000 노출
EXPOSE 8000

# uvicorn으로 FastAPI 앱 실행 (main.py의 app 객체를 호스팅)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]