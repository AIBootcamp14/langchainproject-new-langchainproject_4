# Dockerfile

# --- 1단계: 빌드 환경 설정 (Build Stage) ---
# 안정적인 Python 3.11.9 버전을 사용하고 빌드에 필요한 도구를 설치합니다.
FROM python:3.11.9-slim-bookworm as builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치 (git은 종속성 필요, curl은 헬스체크/다운로드 용)
# 설치 후 캐시를 삭제하여 이미지 크기를 줄입니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
# pip 캐시 사용 안함 (--no-cache-dir) 으로 이미지 크기 최소화
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 2단계: 최종 실행 환경 구성 (Final Stage) ---
# 최종 실행 환경은 빌드 도구가 없는 깨끗한 이미지로 시작하여 이미지 크기를 최소화합니다.
FROM python:3.11.9-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 경로 설정: /app/src 디렉토리를 모듈 경로에 포함시켜 절대 경로로 임포트 가능하게 함
ENV PYTHONPATH /app
# 출력 버퍼링 비활성화 (로그 실시간 확인에 유용)
ENV PYTHONUNBUFFERED 1

# 빌드 단계에서 설치된 라이브러리 및 실행 파일 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 코드 복사 (requirements.txt는 이미 설치됨)
COPY src/ src/

# 데이터 디렉토리 생성 (RAG 시스템/DB/로그 저장을 위해 필수)
RUN mkdir -p /app/data/chroma_db /app/data/raw /app/data/processed /app/logs

# FastAPI의 기본 포트 8080 노출 (docker-compose와 포트 통일)
EXPOSE 8080
# Streamlit 포트 노출 (프론트엔드용)
EXPOSE 8501

# 헬스체크 설정 (FastAPI 포트 8080 사용)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# 기본 실행 명령: uvicorn으로 FastAPI 앱 실행
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]