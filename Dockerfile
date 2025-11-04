# --- 1단계: 빌드 환경 설정 (Build Stage) ---
FROM python:3.11.9-slim-bookworm as builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치 (git, curl, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .

# --- FIX 1: pytest와 coverage 설치 추가 ---
# 테스트 실행을 위해 명시적으로 pytest와 coverage를 설치합니다.
# 테스트 후 주석 풀기
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install pytest coverage

# --------------------------------------------------------------
# 테스트 후 삭제할 코드
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install pytest coverage

# --------------------------------------------------------------

# --- 2단계: 최종 실행 환경 구성 (Final Stage) ---
FROM python:3.11.9-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

ENV PYTHONPATH /app
ENV PYTHONUNBUFFERED 1

# 빌드 단계에서 설치된 라이브러리 및 실행 파일 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- FIX 2: start_api.py 파일 복사 추가 ---
# 최상위 디렉토리에 있는 start_api.py 파일을 /app 디렉토리로 복사합니다.
COPY start_api.py . 

# --- 🚨 핵심 수정: vector_database.py 파일을 명시적으로 먼저 복사하여 캐시 무시 ---
COPY src/modules/vector_database.py src/modules/vector_database.py

# 소스 코드 복사 (나머지 파일들)
COPY src/ src/

# --- FIX 3: data/tests 폴더 경로에 맞게 복사 ---
# tests/ 폴더가 data/tests/에 있으므로, 이 경로의 파일을 컨테이너의 tests/로 복사합니다.
COPY data/tests/ tests/ 

# 데이터 디렉토리 생성 (RAG 시스템/DB/로그 저장을 위해 필수)
RUN mkdir -p /app/data/chroma_db /app/data/raw /app/data/processed /app/logs

# FastAPI와 Streamlit 포트 노출
# 💡 8000으로 통일
EXPOSE 8000
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1 # 💡 8000으로 통일

# 기본 실행 명령: uvicorn으로 FastAPI 앱 실행
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] # 💡 8000으로 통일